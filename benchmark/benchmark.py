"""
Automated benchmarking of every trained model on the CPU.

For each checkpoint in --ckpt-dir (teacher, baselines, distilled students, NAS model):
  * accuracy (top-1 / top-5; taken from the checkpoint, or re-measured with --eval)
  * parameters, PyTorch weight size, ONNX file size
  * PyTorch CPU latency  (mean / p50 / p95, ms) and throughput
  * ONNX Runtime CPU latency (mean / p50 / p95, ms) and throughput
  * peak inference memory for both runtimes (RSS increase, MB)
Each runtime is measured in its own freshly spawned process for clean memory numbers.

Outputs
  results/benchmark.csv        one row per model x runtime x batch size
  results/kd_vs_baseline.csv   KD gain per student, compression / speed-up vs. the teacher

Example
  python -m benchmark.benchmark --threads 1 --runs 100
  python -m benchmark.benchmark --eval            # also re-evaluate accuracy on the test sets
"""
import argparse
import glob
import multiprocessing as mp
import os

import pandas as pd


def _child(q, kind, ckpt_path, onnx_path, size, batch, threads, runs):
    try:
        import torch
        from common.profiling import onnx_cpu_latency, torch_cpu_latency
        if kind == "pytorch":
            from common.utils import load_checkpoint
            from models.registry import model_from_checkpoint
            model = model_from_checkpoint(load_checkpoint(ckpt_path))
            r = torch_cpu_latency(model, size, batch, warmup=10, runs=runs, threads=threads)
        else:
            r = onnx_cpu_latency(onnx_path, size, batch, warmup=10, runs=runs, threads=threads)
        q.put(("ok", r))
    except Exception as e:
        q.put(("err", repr(e)))


def measure(kind, ckpt_path, onnx_path, size, batch, threads, runs):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, kind, ckpt_path, onnx_path, size, batch, threads, runs))
    p.start()
    status, r = q.get(timeout=1800)
    p.join()
    if status != "ok":
        print(f"  ! {kind} failed: {r}")
        return None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (1 = single-core edge setting)")
    ap.add_argument("--batch-sizes", default="1", help="comma list, e.g. 1,32")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--eval", action="store_true", help="re-measure test accuracy (PyTorch, CPU/GPU)")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    from common.utils import count_params, file_size_mb, load_checkpoint, state_dict_size_mb
    from models.registry import model_from_checkpoint

    rows = []
    batches = [int(b) for b in args.batch_sizes.split(",")]
    for ckpt_path in sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt"))):
        ckpt = load_checkpoint(ckpt_path)
        run = ckpt.get("run", os.path.splitext(os.path.basename(ckpt_path))[0])
        model = model_from_checkpoint(ckpt)
        acc, top5 = ckpt.get("acc"), ckpt.get("top5")
        if args.eval:
            from common.data import get_loaders
            from common.engine import evaluate
            from common.utils import get_device
            dev = get_device()
            _, _, test = get_loaders(ckpt["dataset"], args.data, 256, 2, fake=args.fake)
            ev = evaluate(model.to(dev), test, dev, ckpt["input_size"])
            acc, top5 = ev["acc"], ev["top5"]
            model = model.cpu()
        onnx_path = os.path.join(args.onnx_dir, f"{run}.onnx")
        has_onnx = os.path.exists(onnx_path)
        base = {"model": run, "arch": ckpt["arch"], "method": ckpt.get("method", ""), "dataset": ckpt["dataset"],
                "input_size": ckpt["input_size"], "top1_acc": acc, "top5_acc": top5,
                "params_m": count_params(model) / 1e6, "pytorch_size_mb": state_dict_size_mb(model),
                "onnx_size_mb": file_size_mb(onnx_path) if has_onnx else None}
        print(f"{run}: acc={acc} params={base['params_m']:.2f}M")
        for b in batches:
            for kind in ["pytorch", "onnxruntime"]:
                if kind == "onnxruntime" and not has_onnx:
                    continue
                r = measure(kind, ckpt_path, onnx_path, ckpt["input_size"], b, args.threads, args.runs)
                if r is None:
                    continue
                rows.append({**base, "runtime": kind, "batch_size": b, "threads": args.threads, **r})
                print(f"   {kind:12s} bs={b:<3d} {r['latency_mean_ms']:.3f} ms (p95 {r['latency_p95_ms']:.3f}) "
                      f"{r['throughput_img_s']:.0f} img/s peak_mem={r['peak_mem_mb']:.1f} MB")

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "benchmark.csv"), index=False)
    print(f"\nSaved {len(df)} rows -> {args.out}/benchmark.csv")

    # ---- KD vs. baseline comparison (CIFAR-100 students) ---------------------
    if df.empty:
        return
    b1 = df[(df.batch_size == batches[0])]
    pt = b1[b1.runtime == "pytorch"].set_index("model")
    ox = b1[b1.runtime == "onnxruntime"].set_index("model")
    teacher = pt[pt.arch == "resnet50"]
    comp = []
    for arch in ["mobilenetv2", "squeezenet", "efficientnet_lite0"]:
        bl, kd = f"baseline_{arch}", f"kd_{arch}"
        if bl not in pt.index and kd not in pt.index:
            continue
        row = {"student": arch,
               "baseline_acc": pt.loc[bl, "top1_acc"] if bl in pt.index else None,
               "kd_acc": pt.loc[kd, "top1_acc"] if kd in pt.index else None}
        if row["baseline_acc"] is not None and row["kd_acc"] is not None:
            row["kd_gain_pts"] = row["kd_acc"] - row["baseline_acc"]
        ref = kd if kd in pt.index else bl
        row.update(params_m=pt.loc[ref, "params_m"], size_mb=pt.loc[ref, "pytorch_size_mb"],
                   pytorch_latency_ms=pt.loc[ref, "latency_mean_ms"],
                   onnx_latency_ms=ox.loc[ref, "latency_mean_ms"] if ref in ox.index else None)
        if len(teacher):
            t = teacher.iloc[0]
            row.update(teacher_acc=t["top1_acc"], compression_x=t["pytorch_size_mb"] / row["size_mb"],
                       speedup_vs_teacher_x=t["latency_mean_ms"] / row["pytorch_latency_ms"])
        comp.append(row)
    if comp:
        pd.DataFrame(comp).to_csv(os.path.join(args.out, "kd_vs_baseline.csv"), index=False)
        print(pd.DataFrame(comp).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
