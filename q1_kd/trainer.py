"""Shared CLI arguments and training driver for teacher / baseline / KD scripts."""
import argparse
import os
import time

import torch

from common.data import NUM_CLASSES, get_loaders
from common.engine import evaluate, fit
from common.utils import (CSVLogger, count_params, get_device, load_checkpoint, save_checkpoint,
                          save_json, set_seed, state_dict_size_mb)


def base_parser(desc, epochs=200, lr=0.1):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    p.add_argument("--data", default="./data", help="dataset root (auto-downloaded)")
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=1, help="warm-up epochs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fake", action="store_true", help="use random FakeData (smoke test, no download)")
    p.add_argument("--max-steps", type=int, default=None, help="limit iterations/epoch (smoke test)")
    return p


def setup(args):
    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    args.amp = bool(args.amp and device.type == "cuda")
    train_loader, _, test_loader = get_loaders(args.dataset, args.data, args.batch_size, args.workers,
                                               val_size=0, seed=args.seed, fake=args.fake)
    print(f"device={device} amp={args.amp} train_batches={len(train_loader)} test_batches={len(test_loader)}")
    return device, train_loader, test_loader


def train_and_save(run_name, arch, model, step_fn, params_module, train_loader, test_loader, device, args,
                   extra_meta=None, trainable_params=None):
    """Train with `step_fn`, keep the best checkpoint on the test set, write logs + summary json."""
    ckpt_path = os.path.join(args.ckpt_dir, f"{run_name}.pt")
    logger = CSVLogger(os.path.join(args.results_dir, "logs", f"{run_name}.csv"))
    meta = {"arch": arch, "num_classes": NUM_CLASSES[args.dataset], "dataset": args.dataset,
            "input_size": 32, "run": run_name, **(extra_meta or {})}
    best_state = {"acc": -1}

    def ckpt_cb(epoch, ev, improved):
        if improved:
            best_state.update(acc=ev["acc"], top5=ev["top5"], epoch=epoch)
            save_checkpoint(ckpt_path, model, {**meta, "acc": ev["acc"], "top5": ev["top5"], "epoch": epoch})

    t0 = time.time()
    best, history = fit(model, step_fn, params_module, train_loader, test_loader, device, args.epochs,
                        args.lr, args.wd, logger=logger, amp=args.amp, ckpt_cb=ckpt_cb,
                        warmup_epochs=args.warmup, extra_params=trainable_params,
                        log_prefix=f"[{run_name}] ", max_steps=args.max_steps)
    summary = {
        "run": run_name, "arch": arch, "best_acc": best_state["acc"], "best_top5": best_state.get("top5"),
        "best_epoch": best_state.get("epoch"), "final_acc": history[-1]["eval_acc"],
        "params_M": count_params(model) / 1e6, "size_mb": state_dict_size_mb(model),
        "train_time_min": (time.time() - t0) / 60, "epochs": args.epochs, "checkpoint": ckpt_path,
        **(extra_meta or {}),
    }
    save_json(summary, os.path.join(args.results_dir, "q1", f"{run_name}.json"))
    print(f"[{run_name}] done: best acc {best_state['acc']:.2f}% -> {ckpt_path}")
    return summary


def load_teacher(path, device):
    from models.registry import model_from_checkpoint
    ckpt = load_checkpoint(path)
    teacher = model_from_checkpoint(ckpt).to(device)
    print(f"Loaded teacher {ckpt['arch']} ({ckpt.get('acc', float('nan')):.2f}% top-1) from {path}")
    return teacher
