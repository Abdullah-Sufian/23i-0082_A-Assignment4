# Edge Computing & Intelligent Systems — Assignment 4

**Q1:** Knowledge distillation from ResNet-50 into MobileNetV2, SqueezeNet and EfficientNet-Lite0 on CIFAR-100.
**Q2:** Hardware-aware neural architecture search for edge CNNs on CIFAR-10.
**Both:** ONNX export, ONNX Runtime CPU inference and automated benchmarking.

## Repository layout

```
common/            data loaders, training engine, CPU profiling (latency / size / peak memory), class names
models/            teacher (ResNet-50), students (MobileNetV2, SqueezeNet, EfficientNet-Lite0), registry
q1_kd/
  kd.py            KD framework: temperature softmax, soft-target KL, hard+soft loss, FitNet/AT feature loss
  train_teacher.py teacher training (ResNet-50, CIFAR-100)
  train_baseline.py  students WITHOUT distillation (baseline pipeline)
  train_kd.py      students WITH distillation
q2_nas/
  search_space.py  searchable space: depth, channels, kernel size, block type (conv / depthwise-separable / MBConv),
                   activation, input resolution; random sampling, mutation, Optuna define-by-run
  generator.py     architecture generator: config dict -> nn.Module
  hw_profiler.py   CPU latency, storage size, peak inference memory, constraint checker
  evaluator.py     automated proxy training + validation for each candidate, hardware-aware score
  search.py        regularized evolution  OR  Optuna TPE search (--strategy)
  train_best.py    retrains the best architecture from scratch on full CIFAR-10
deploy/
  pipeline.py      unified inference pipeline (PyTorch or ONNX Runtime backend, same preprocessing)
  infer.py         inference CLI for every model (images / folder / CIFAR samples / full test eval)
  export_onnx.py   ONNX export + checker + numerical PyTorch-vs-ORT verification
  onnx_infer.py    CPU-only ONNX Runtime inference + latency
benchmark/benchmark.py   automated CPU benchmark of all models -> results/benchmark.csv, kd_vs_baseline.csv
tools/make_report_assets.py  plots + markdown tables for the report -> results/plots, results/summary.md
run_all.sh         whole pipeline (MODE=smoke | quick | full)
colab_runner.ipynb same pipeline as notebook cells for Google Colab
checkpoints/  onnx/  results/   generated artefacts (trained models, exported models, CSV/JSON results)
docs/REPORT.md     technical report source (export to PDF)
```

## Deliverable map

| # | Deliverable | Where |
|---|---|---|
| 1 | Teacher model | `q1_kd/train_teacher.py`, `models/teacher.py`, `checkpoints/teacher_resnet50.pt` |
| 2 | Student architectures | `models/students.py` |
| 3 | Baseline students | `q1_kd/train_baseline.py`, `checkpoints/baseline_*.pt` |
| 4 | Knowledge distillation | `q1_kd/kd.py`, `q1_kd/train_kd.py`, `checkpoints/kd_*.pt` |
| 5 | NAS search space | `q2_nas/search_space.py`, `q2_nas/generator.py` |
| 6 | NAS algorithm | `q2_nas/search.py` (evolution + Optuna), `q2_nas/evaluator.py`, `q2_nas/hw_profiler.py` |
| 7 | NAS results | `results/nas/best_arch.json`, `results/nas/search_history.csv/.json` |
| 8 | Optimized architecture | `q2_nas/train_best.py`, `checkpoints/nas_best.pt` |
| 9 | Inference pipeline | `deploy/pipeline.py`, `deploy/infer.py` |
| 10 | ONNX deployment | `deploy/export_onnx.py`, `deploy/onnx_infer.py`, `onnx/*.onnx` |
| 11 | Benchmarking | `benchmark/benchmark.py`, `results/benchmark.csv`, `results/kd_vs_baseline.csv` |
| 12 | Technical report | `docs/REPORT.md` → PDF |
| 13 | README | this file |

## Installation

Python 3.9+. A CUDA GPU is strongly recommended for training. All inference and benchmarking runs on the CPU.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or the CPU wheel
pip install -r requirements.txt
```

CIFAR-10 and CIFAR-100 download automatically into `./data` the first time you run a script.

**Sanity check (2 minutes, CPU, no download):** `MODE=smoke bash run_all.sh`
This runs every stage on random FakeData so you can confirm the install works.
Delete `checkpoints/ onnx/ results/` afterwards.

## Step-by-step execution

Run every command from the repository root. Add `--amp` on a CUDA GPU for mixed precision. Time estimates are for one T4 in quick mode.

### Question 1: knowledge distillation (CIFAR-100)

```bash
# 1. Teacher. Quick: ImageNet-initialised, ~30 min. Full: from scratch, --epochs 200.
python -m q1_kd.train_teacher --pretrained --epochs 30 --lr 0.05 --amp

# 2. Baseline students, plain cross-entropy, no KD
python -m q1_kd.train_baseline --arch all --epochs 30 --amp

# 3. Distilled students: soft targets (T=4), hard+soft mix (alpha=0.9), FitNet feature hints (beta=1)
python -m q1_kd.train_kd --arch all --epochs 30 --teacher checkpoints/teacher_resnet50.pt \
       --T 4 --alpha 0.9 --beta 1.0 --feat-loss fitnet --amp
```

Useful ablations: `--beta 0 --name-suffix _logitonly` (no feature loss), `--alpha 0` (features only),
`--feat-loss attention --beta 1000` (attention transfer), and `--T 2/8` (temperature).

### Question 2: hardware-aware NAS (CIFAR-10)

```bash
# 4. Search. Each candidate is profiled on the CPU first. Violators are rejected without training.
python -m q2_nas.search --strategy evolution --budget 40 --population 10 --proxy-epochs 5 \
       --max-latency-ms 8 --max-size-mb 2 --max-peak-mem-mb 64 --amp
#    Bayesian alternative:  --strategy optuna

# 5. Retrain the best discovered architecture from scratch
python -m q2_nas.train_best --epochs 60 --amp
```

The constraints apply to CPU latency at batch size 1 on one thread (`--threads`). Tune them to your target device. If nothing passes, relax them.

### Deployment and benchmarking

```bash
python -m deploy.export_onnx --include-teacher              # onnx/*.onnx + metadata .json, verified vs PyTorch
python -m deploy.infer --all --cifar-samples 5              # unified pipeline, PyTorch backend
python -m deploy.onnx_infer --all --cifar-samples 5         # ONNX Runtime, CPU
python -m deploy.onnx_infer --model onnx/kd_mobilenetv2.onnx --images my_cat.jpg
python -m deploy.onnx_infer --model onnx/nas_best.onnx --eval   # test accuracy through ORT
python -m benchmark.benchmark --threads 1 --runs 100        # results/benchmark.csv, kd_vs_baseline.csv
python -m tools.make_report_assets                          # results/plots/*.png, results/summary.md
```

Or run everything at once: `bash run_all.sh` (quick) / `MODE=full bash run_all.sh`.

## Outputs

| File | Content |
|---|---|
| `results/logs/<run>.csv` | per-epoch loss, CE / KD / feature loss components, test accuracy |
| `results/q1/<run>.json` | best accuracy, params, size, training time per Q1 run |
| `results/nas/search_history.csv/.json` | every NAS candidate: config, latency, size, peak memory, feasibility, reject reason, accuracy, score |
| `results/nas/best_arch.json` | selected architecture, constraints, search space |
| `results/nas/best_final.json` | best architecture after full training (test accuracy + hardware metrics) |
| `results/onnx_export.csv` | ONNX size and max PyTorch-vs-ORT output difference |
| `results/benchmark.csv` | accuracy, size, latency (mean/p50/p95), throughput, peak memory for PyTorch and ORT |

## Notes

- `checkpoints/teacher_resnet50.pt` is about 90 MB. Use **Git LFS** (`git lfs install && git lfs track "*.pt" "*.onnx"`). `.gitattributes` is already set up.
- Peak memory is the RSS increase during inference, measured in a freshly spawned process. It needs `psutil` and shows as `NaN` when unsupported.
- Inputs are 32×32 (NAS models use their searched resolution, 24/28/32). The inference pipeline resizes any image automatically.
