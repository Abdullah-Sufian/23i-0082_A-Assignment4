"""
Generate figures and markdown tables for the technical report from the CSV/JSON results.

Reads   results/logs/*.csv, results/benchmark.csv, results/kd_vs_baseline.csv,
        results/nas/search_history.csv, results/nas/best_final.json
Writes  results/plots/*.png and results/summary.md

Example
  python -m tools.make_report_assets
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# fixed categorical order (identity colour follows the entity, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6", "#7d7d7d"]
GREY = "#9a9a94"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "grid.linewidth": 0.8, "axes.edgecolor": "#9a9a94",
                     "font.size": 10, "lines.linewidth": 2, "figure.dpi": 130, "savefig.bbox": "tight"})
STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def md_table(df, floatfmt=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    return "\n".join([head, sep, *rows])


def plot_curves(res, out, md):
    logs = {os.path.splitext(os.path.basename(p))[0]: pd.read_csv(p)
            for p in sorted(glob.glob(os.path.join(res, "logs", "*.csv")))}
    if not logs:
        return
    groups = {"Teacher (ResNet-50, CIFAR-100)": [k for k in logs if k.startswith("teacher")],
              "Students: baseline vs KD (CIFAR-100)": [k for k in logs if k.startswith(("baseline_", "kd_"))],
              "NAS best model (CIFAR-10)": [k for k in logs if k.startswith("nas")]}
    for title, keys in groups.items():
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in keys:
            arch = next((s for s in STUDENTS if s in k), None)
            color = SERIES[STUDENTS.index(arch)] if arch else SERIES[0]
            ls = "--" if k.startswith("baseline_") else "-"
            ax.plot(logs[k]["epoch"], logs[k]["eval_acc"], ls, color=color, label=k)
        ax.set(title=title, xlabel="epoch", ylabel="test top-1 accuracy (%)")
        ax.legend(frameon=False, fontsize=8)
        name = "curves_" + title.split()[0].lower().strip(":") + ".png"
        fig.savefig(os.path.join(out, name))
        plt.close(fig)
        md.append(f"![{title}](plots/{name})\n")


def plot_kd(res, out, md):
    p = os.path.join(res, "kd_vs_baseline.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    md += ["## Q1 - Knowledge distillation vs. baseline\n", md_table(df), ""]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    w = 0.38
    ax.bar([i - w / 2 for i in x], df["baseline_acc"], w - 0.02, color=SERIES[0], label="baseline (no KD)")
    ax.bar([i + w / 2 for i in x], df["kd_acc"], w - 0.02, color=SERIES[1], label="distilled (KD)")
    if "teacher_acc" in df and df["teacher_acc"].notna().any():
        ax.axhline(df["teacher_acc"].iloc[0], color=GREY, ls=":", lw=1.5, label="ResNet-50 teacher")
    for i, r in df.iterrows():
        if pd.notna(r.get("kd_gain_pts")):
            ax.annotate(f"{r['kd_gain_pts']:+.2f}", (i + w / 2, r["kd_acc"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x), df["student"])
    ax.set(ylabel="CIFAR-100 top-1 accuracy (%)", title="Effect of knowledge distillation")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "kd_vs_baseline.png"))
    plt.close(fig)
    md.append("![KD vs baseline](plots/kd_vs_baseline.png)\n")


def plot_benchmark(res, out, md):
    p = os.path.join(res, "benchmark.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    cols = ["model", "runtime", "batch_size", "top1_acc", "params_m", "pytorch_size_mb", "onnx_size_mb",
            "latency_mean_ms", "latency_p95_ms", "throughput_img_s", "peak_mem_mb"]
    md += ["## Benchmark (CPU)\n", md_table(df[[c for c in cols if c in df]]), ""]
    b = df[df.batch_size == df.batch_size.min()]
    for ds, title in [("cifar100", "Q1 models (CIFAR-100)"), ("cifar10", "NAS model (CIFAR-10)")]:
        d = b[b.dataset == ds]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for j, rt in enumerate(["pytorch", "onnxruntime"]):
            s = d[d.runtime == rt]
            ax.scatter(s["latency_mean_ms"], s["top1_acc"], s=60, color=SERIES[j], edgecolor="white",
                       linewidth=1.5, label=rt, zorder=3)
            for _, r in (s.iterrows() if rt == "pytorch" else []):  # label once per model
                ax.annotate(r["model"], (r["latency_mean_ms"], r["top1_acc"]), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set(xlabel="CPU latency, batch 1 (ms, log scale)", ylabel="top-1 accuracy (%)",
               title=f"Accuracy vs. CPU latency - {title}")
        ax.legend(frameon=False, fontsize=8)
        name = f"acc_vs_latency_{ds}.png"
        fig.savefig(os.path.join(out, name))
        plt.close(fig)
        md.append(f"![acc vs latency]({'plots/' + name})\n")


def plot_nas(res, out, md):
    p = os.path.join(res, "nas", "search_history.csv")
    if not os.path.exists(p):
        return
    h = pd.read_csv(p)
    feas, rej = h[h.feasible], h[~h.feasible]
    md += ["## Q2 - NAS search\n",
           f"Evaluated **{len(h)}** architectures: **{len(feas)}** feasible, **{len(rej)}** rejected by "
           f"deployment constraints.\n"]
    cols = ["index", "arch", "latency_mean_ms", "size_mb", "peak_mem_mb", "params_m", "val_acc", "score"]
    md += ["Top-10 feasible architectures:\n", md_table(feas.sort_values("score", ascending=False)[cols].head(10)), ""]
    if len(rej):
        md += ["Rejected architectures (first 10):\n",
               md_table(rej[["index", "arch", "latency_mean_ms", "size_mb", "reject_reason"]].head(10)), ""]
    # latency vs accuracy of feasible candidates; rejected shown as a rug along the x-axis
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(feas["latency_mean_ms"], feas["val_acc"], s=40, color=SERIES[0], edgecolor="white",
               linewidth=1.5, label="feasible", zorder=3)
    if len(rej):
        y0 = feas["val_acc"].min() if len(feas) else 0
        ax.scatter(rej["latency_mean_ms"], [y0] * len(rej), marker="|", s=120, color=GREY,
                   label="rejected (constraint violated)")
    if len(feas):
        best = feas.loc[feas["score"].idxmax()]
        ax.scatter([best["latency_mean_ms"]], [best["val_acc"]], s=160, marker="*", color=SERIES[1],
                   edgecolor="white", zorder=4, label="selected")
    try:
        lim = json.load(open(os.path.join(res, "nas", "best_arch.json")))["constraints"]["max_latency_ms"]
        ax.axvline(lim, color=GREY, ls=":", lw=1.5, label=f"latency budget {lim} ms")
    except Exception:
        pass
    ax.set(xlabel="CPU latency, batch 1 (ms)", ylabel="proxy validation accuracy (%)",
           title="NAS candidates: accuracy vs. measured latency")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "nas_scatter.png"))
    plt.close(fig)
    # search progress
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sc = h["score"].fillna(float("-inf")).cummax().where(lambda s: s > float("-inf"))
    ax.plot(h["index"], sc, color=SERIES[0], label="best score so far")
    ax.scatter(feas["index"], feas["score"], s=20, color=SERIES[0], alpha=0.5, label="candidate score")
    ax.set(xlabel="architectures evaluated", ylabel="hardware-aware score", title="NAS search progress")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "nas_progress.png"))
    plt.close(fig)
    md += ["![NAS scatter](plots/nas_scatter.png)\n", "![NAS progress](plots/nas_progress.png)\n"]
    fp = os.path.join(res, "nas", "best_final.json")
    if os.path.exists(fp):
        f = json.load(open(fp))
        md += ["### Best architecture trained from scratch\n",
               f"`{f['arch']}`\n",
               md_table(pd.DataFrame([{k: f.get(k) for k in ["test_acc", "search_val_acc", "params_m", "size_mb",
                                                             "latency_mean_ms", "latency_p95_ms", "peak_mem_mb",
                                                             "epochs"]}])), ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    out = os.path.join(a.results, "plots")
    os.makedirs(out, exist_ok=True)
    md = ["# Experimental results summary\n"]
    plot_kd(a.results, out, md)
    plot_benchmark(a.results, out, md)
    plot_nas(a.results, out, md)
    md.append("## Training curves\n")
    plot_curves(a.results, out, md)
    with open(os.path.join(a.results, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {a.results}/summary.md and plots in {out}/")


if __name__ == "__main__":
    main()
