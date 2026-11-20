"""
Load the best architecture discovered by the search and train it FROM SCRATCH on
the full CIFAR-10 training set, then evaluate on the CIFAR-10 test set.

Example
  python -m q2_nas.train_best --arch-json results/nas/best_arch.json --epochs 100 --amp
"""
import argparse
import os
import time

from common.data import get_loaders
from common.engine import evaluate, fit, make_ce_step
from common.utils import CSVLogger, count_params, get_device, load_json, save_checkpoint, save_json, set_seed
from q2_nas.generator import build_from_config
from q2_nas.hw_profiler import profile_config
from q2_nas.search_space import describe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch-json", default="results/nas/best_arch.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", default="checkpoints/nas_best.pt")
    p.add_argument("--out", default="results/nas")
    p.add_argument("--logs-dir", default="results/logs")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    args.amp = bool(args.amp and device.type == "cuda")
    info = load_json(args.arch_json)
    config = info["config"]
    res = config["resolution"]
    print(f"Training best architecture from scratch: {describe(config)}")

    train_loader, _, test_loader = get_loaders("cifar10", args.data, args.batch_size, args.workers,
                                               val_size=0, seed=args.seed, fake=args.fake)
    model = build_from_config(config, 10).to(device)  # fresh random init = from scratch
    meta = {"arch": "nas", "num_classes": 10, "dataset": "cifar10", "input_size": res,
            "nas_config": config, "run": "nas_best", "method": "nas"}

    def ckpt_cb(epoch, ev, improved):
        if improved:
            save_checkpoint(args.ckpt, model, {**meta, "acc": ev["acc"], "top5": ev["top5"], "epoch": epoch})

    t0 = time.time()
    best, hist = fit(model, make_ce_step(model), model, train_loader, test_loader, device, args.epochs,
                     args.lr, args.wd, logger=CSVLogger(os.path.join(args.logs_dir, "nas_best.csv")),
                     amp=args.amp, input_size=res, ckpt_cb=ckpt_cb, log_prefix="[nas_best] ",
                     max_steps=args.max_steps)
    prof = profile_config(config, 10, threads=1)
    summary = {"arch": describe(config), "config": config, "test_acc": best, "epochs": args.epochs,
               "params_m": count_params(model) / 1e6, "train_time_min": (time.time() - t0) / 60,
               "search_val_acc": info["best"].get("val_acc"), **prof, "checkpoint": args.ckpt}
    save_json(summary, os.path.join(args.out, "best_final.json"))
    print(f"Final test accuracy {best:.2f}% | latency {prof['latency_mean_ms']:.2f}ms | "
          f"size {prof['size_mb']:.3f}MB -> {args.ckpt}")


if __name__ == "__main__":
    main()
