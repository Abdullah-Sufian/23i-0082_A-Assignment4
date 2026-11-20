"""
Hardware-aware Neural Architecture Search on CIFAR-10.

Two search strategies over the same search space:
  --strategy evolution : regularized (aging) evolution, Real et al. 2019
        1. fill a population with random *feasible* architectures
        2. repeat: pick `sample_size` members at random, take the best (tournament),
           mutate it, evaluate the child, add it to the population and remove the
           OLDEST member (aging keeps the search exploring)
  --strategy optuna    : Bayesian optimisation with Optuna's TPE sampler;
        constraint violators are reported to Optuna as pruned trials

Outputs (in --out, default results/nas/):
  search_history.csv / search_history.json   every candidate (feasible or rejected) with all metrics
  best_arch.json                             best feasible architecture + its metrics
  optuna_trials.csv                          (optuna only) Optuna's own trial table

Example
  python -m q2_nas.search --strategy evolution --budget 40 --population 10 --proxy-epochs 5 --amp
  python -m q2_nas.search --strategy optuna --budget 40 --max-latency-ms 5 --max-size-mb 1.0
"""
import argparse
import collections
import json
import os
import random
import time

import pandas as pd

from common.utils import ensure_dir, get_device, save_json, set_seed
from q2_nas.evaluator import CandidateEvaluator
from q2_nas.hw_profiler import Constraints
from q2_nas.search_space import SPACE, mutate, sample_config, suggest_config


def parse():
    p = argparse.ArgumentParser(description="Hardware-aware NAS for edge devices (CIFAR-10)")
    p.add_argument("--strategy", default="evolution", choices=["evolution", "optuna"])
    p.add_argument("--budget", type=int, default=40, help="number of architectures to evaluate (incl. rejected)")
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--sample-size", type=int, default=4, help="tournament size")
    # proxy training
    p.add_argument("--proxy-epochs", type=int, default=5)
    p.add_argument("--train-subset", type=int, default=15000, help="training images per candidate")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    # deployment constraints
    p.add_argument("--max-latency-ms", type=float, default=8.0)
    p.add_argument("--max-size-mb", type=float, default=2.0)
    p.add_argument("--max-peak-mem-mb", type=float, default=64.0)
    p.add_argument("--max-params-m", type=float, default=1e9)
    p.add_argument("--latency-weight", type=float, default=-0.07, help="w in acc*(lat/target)^w")
    # profiling
    p.add_argument("--threads", type=int, default=1, help="CPU threads for latency measurement")
    p.add_argument("--latency-runs", type=int, default=50)
    p.add_argument("--inprocess-profiling", action="store_true", help="profile in this process (faster, noisier memory)")
    # misc
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/nas")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def save_history(ev, args, out):
    rows = []
    for r in ev.history:
        row = {k: v for k, v in r.items() if k != "config"}
        row["config_json"] = json.dumps(r["config"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "search_history.csv"), index=False)
    save_json(ev.history, os.path.join(out, "search_history.json"))
    best = ev.best()
    if best:
        save_json({"strategy": args.strategy, "constraints": ev.constraints.as_dict(),
                   "search_space": SPACE, "best": best, "config": best["config"]},
                  os.path.join(out, "best_arch.json"))


def run_evolution(ev, args, out):
    rng = random.Random(args.seed)
    population = collections.deque()
    # 1) initial random population (only feasible architectures enter it)
    while len(population) < args.population and len(ev.history) < args.budget:
        rec = ev.evaluate(sample_config(rng), origin="random")
        if rec["feasible"]:
            population.append(rec)
        save_history(ev, args, out)
    if not population:
        print("No feasible architecture found in the initial population - relax the constraints.")
        return
    # 2) aging evolution
    while len(ev.history) < args.budget:
        sample = rng.sample(list(population), min(args.sample_size, len(population)))
        parent = max(sample, key=lambda r: r["score"])
        before = len(ev.history)
        child = ev.evaluate(mutate(parent["config"], rng), origin=f"mutation_of_{parent['id']}")
        if len(ev.history) == before:  # duplicate (cached) -> mutate again next round
            continue
        if child["feasible"]:
            population.append(child)
            population.popleft()  # remove oldest
        save_history(ev, args, out)


def run_optuna(ev, args, out):
    import optuna
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=min(10, args.budget // 3 or 1))
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="edge_nas")

    def objective(trial):
        rec = ev.evaluate(suggest_config(trial), origin=f"optuna_trial_{trial.number}")
        for k in ("latency_mean_ms", "size_mb", "peak_mem_mb", "params_m", "val_acc"):
            trial.set_user_attr(k, rec.get(k))
        save_history(ev, args, out)
        if not rec["feasible"]:
            trial.set_user_attr("reject_reason", rec["reject_reason"])
            raise optuna.TrialPruned(rec["reject_reason"])  # constraint violation
        return rec["score"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.budget)
    study.trials_dataframe().to_csv(os.path.join(out, "optuna_trials.csv"), index=False)


def main():
    args = parse()
    set_seed(args.seed)
    device = get_device(args.device)
    args.amp = bool(args.amp and device.type == "cuda")
    out = ensure_dir(args.out)
    cons = Constraints(args.max_latency_ms, args.max_size_mb, args.max_peak_mem_mb, args.max_params_m)
    print(f"Strategy={args.strategy} budget={args.budget} device={device} constraints={cons.as_dict()}")
    save_json(vars(args), os.path.join(out, "search_args.json"))
    ev = CandidateEvaluator(args, cons, device)
    t0 = time.time()
    (run_evolution if args.strategy == "evolution" else run_optuna)(ev, args, out)
    save_history(ev, args, out)
    best = ev.best()
    n_rej = sum(not r["feasible"] for r in ev.history)
    print(f"\nSearch finished in {(time.time() - t0) / 60:.1f} min: {len(ev.history)} evaluated, {n_rej} rejected.")
    if best:
        print(f"BEST: {best['arch']}\n  val_acc={best['val_acc']:.2f}% latency={best['latency_mean_ms']:.2f}ms "
              f"size={best['size_mb']:.3f}MB peak_mem={best['peak_mem_mb']:.1f}MB -> {out}/best_arch.json")


if __name__ == "__main__":
    main()
