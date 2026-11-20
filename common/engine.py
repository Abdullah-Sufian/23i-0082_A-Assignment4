"""
Generic training / evaluation engine shared by every training script
(teacher, baseline students, distilled students, NAS candidates, NAS final model).

The training loop is decoupled from the loss through `step_fn(x, y) -> (loss, logits, parts)`
so the same loop serves plain cross-entropy training and knowledge distillation.
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_optimizer(params, lr=0.1, weight_decay=5e-4, momentum=0.9, name="sgd"):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=1):
    """Linear warm-up followed by cosine decay, stepped every iteration."""
    total = max(1, epochs * steps_per_epoch)
    warm = min(total - 1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def accuracy(logits, target, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.view(-1, 1))
    return [correct[:, :k].any(dim=1).float().sum().item() for k in topk]


def resize_batch(x, size):
    """Resize a batch to `size` x `size` (used for NAS variable input resolution)."""
    if size is None or x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def train_one_epoch(step_fn, params_module, loader, optimizer, scheduler, device,
                    scaler=None, amp=False, input_size=None, grad_clip=None, max_steps=None):
    """Run one epoch. `params_module` is put in train mode (may contain student + regressors)."""
    params_module.train()
    n, loss_sum, correct = 0, 0.0, 0.0
    parts_sum = {}
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        if max_steps and i >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y).to(device, non_blocking=True).long()
        x = resize_batch(x, input_size)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                            enabled=amp):
            loss, logits, parts = step_fn(x, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params_module.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(params_module.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = x.size(0)
        n += bs
        loss_sum += loss.item() * bs
        correct += accuracy(logits.float(), y)[0]
        for k, v in (parts or {}).items():
            parts_sum[k] = parts_sum.get(k, 0.0) + float(v) * bs
    out = {"train_loss": loss_sum / max(1, n), "train_acc": 100.0 * correct / max(1, n),
           "epoch_time_s": time.time() - t0}
    out.update({k: v / max(1, n) for k, v in parts_sum.items()})
    return out


@torch.no_grad()
def evaluate(model, loader, device, input_size=None, amp=False, max_steps=None):
    """Top-1 / Top-5 accuracy and CE loss of `model` on `loader`."""
    model.eval()
    n, loss_sum, c1, c5 = 0, 0.0, 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        if max_steps and i >= max_steps:
            break
        x = resize_batch(x.to(device, non_blocking=True), input_size)
        y = torch.as_tensor(y).to(device).long()
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                            enabled=amp):
            logits = model(x)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        k5 = min(5, logits.size(1))
        a1, a5 = accuracy(logits, y, (1, k5))
        c1 += a1
        c5 += a5
        n += x.size(0)
    return {"acc": 100.0 * c1 / max(1, n), "top5": 100.0 * c5 / max(1, n), "loss": loss_sum / max(1, n)}


def make_ce_step(model, label_smoothing=0.0):
    """Plain supervised step (teacher training, baseline students, NAS)."""
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def step(x, y):
        logits = model(x)
        loss = ce(logits, y)
        return loss, logits, {"ce": loss.item()}

    return step


def fit(model, step_fn, params_module, train_loader, eval_loader, device, epochs, lr, wd,
        logger=None, amp=False, input_size=None, ckpt_cb=None, warmup_epochs=1, optimizer_name="sgd",
        extra_params=None, log_prefix="", max_steps=None):
    """Full training run with per-epoch evaluation; returns best eval accuracy and history."""
    params = list(params_module.parameters()) if extra_params is None else extra_params
    optimizer = build_optimizer(params, lr=lr, weight_decay=wd, name=optimizer_name)
    steps = len(train_loader) if not max_steps else min(len(train_loader), max_steps)
    scheduler = build_scheduler(optimizer, epochs, steps, warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(step_fn, params_module, train_loader, optimizer, scheduler, device,
                             scaler, amp, input_size, max_steps=max_steps)
        ev = evaluate(model, eval_loader, device, input_size, amp)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **tr,
               "eval_acc": ev["acc"], "eval_top5": ev["top5"], "eval_loss": ev["loss"]}
        history.append(row)
        if logger:
            logger.log(row)
        improved = ev["acc"] > best
        best = max(best, ev["acc"])
        if ckpt_cb:
            ckpt_cb(epoch, ev, improved)
        extra = " ".join(f"{k}={tr[k]:.4f}" for k in ("ce", "kd", "feat") if k in tr)
        print(f"{log_prefix}[{epoch:03d}/{epochs}] loss={tr['train_loss']:.4f} train_acc={tr['train_acc']:.2f} "
              f"{extra} | eval_acc={ev['acc']:.2f} top5={ev['top5']:.2f} best={best:.2f} "
              f"({tr['epoch_time_s']:.1f}s)", flush=True)
    return best, history
