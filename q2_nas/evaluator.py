"""
Automated training + validation pipeline for NAS candidates (CIFAR-10).

Every candidate goes through:
  1. hardware profiling on CPU (latency / size / peak memory)
  2. constraint check  -> violators are REJECTED without spending any training time
  3. short proxy training on a subset of the CIFAR-10 training split
  4. validation accuracy on a held-out split of the training set (the test set is
     never seen during search)
  5. hardware-aware score (MnasNet-style soft latency reward):
        score = val_acc * (latency / target_latency) ** w,   w = -0.07
"""
import time

import torch

from common.data import get_loaders
from common.engine import fit, make_ce_step
from q2_nas.generator import build_from_config
from q2_nas.hw_profiler import profile_config
from q2_nas.search_space import config_id, describe, total_depth


class CandidateEvaluator:
    def __init__(self, args, constraints, device):
        self.args = args
        self.constraints = constraints
        self.device = device
        self.cache = {}
        self.history = []
        self.train_loader, self.val_loader, _ = get_loaders(
            "cifar10", args.data, args.batch_size, args.workers, val_size=args.val_size,
            train_subset=args.train_subset, seed=args.seed, fake=args.fake)
        print(f"NAS proxy data: {len(self.train_loader.dataset)} train / {len(self.val_loader.dataset)} val images")

    def score(self, acc, latency):
        target = self.constraints.max_latency_ms
        return acc * (latency / target) ** self.args.latency_weight

    def evaluate(self, config, origin="random"):
        cid = config_id(config)
        if cid in self.cache:                       # never train the same architecture twice
            return self.cache[cid]
        rec = {"id": cid, "index": len(self.history), "origin": origin, "arch": describe(config),
               "resolution": config["resolution"], "num_stages": len(config["stages"]),
               "total_blocks": total_depth(config)}
        t0 = time.time()
        prof = profile_config(config, 10, threads=self.args.threads, runs=self.args.latency_runs,
                              isolated=not self.args.inprocess_profiling)
        rec.update({k: round(v, 4) if isinstance(v, float) else v for k, v in prof.items()})
        ok, reasons = self.constraints.check(prof)
        rec["feasible"] = ok
        rec["reject_reason"] = "; ".join(reasons)
        if ok:
            torch.manual_seed(self.args.seed)
            model = build_from_config(config, 10).to(self.device)
            best, _ = fit(model, make_ce_step(model), model, self.train_loader, self.val_loader, self.device,
                          self.args.proxy_epochs, self.args.lr, self.args.wd, amp=self.args.amp,
                          input_size=config["resolution"], warmup_epochs=0, log_prefix="    ",
                          max_steps=self.args.max_steps)
            rec["val_acc"] = round(best, 3)
            rec["score"] = round(self.score(best, prof["latency_mean_ms"]), 4)
        else:
            rec["val_acc"] = None
            rec["score"] = None
        rec["eval_time_s"] = round(time.time() - t0, 1)
        rec["config"] = config
        self.cache[cid] = rec
        self.history.append(rec)
        status = (f"acc={rec['val_acc']:.2f} score={rec['score']:.3f}" if ok else f"REJECTED ({rec['reject_reason']})")
        print(f"[{rec['index']:03d}] {rec['arch']}\n      lat={prof['latency_mean_ms']:.2f}ms "
              f"size={prof['size_mb']:.3f}MB mem={prof['peak_mem_mb']:.1f}MB params={prof['params_m']:.3f}M -> {status}",
              flush=True)
        return rec

    def best(self):
        feas = [r for r in self.history if r["feasible"]]
        return max(feas, key=lambda r: r["score"]) if feas else None
