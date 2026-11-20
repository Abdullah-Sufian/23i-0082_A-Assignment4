#!/usr/bin/env bash
# End-to-end pipeline for both questions.
#   bash run_all.sh            # quick mode (~2-2.5 h on a single T4/RTX GPU)
#   MODE=full bash run_all.sh  # full-length training recipe (~10+ h)
#   MODE=smoke bash run_all.sh # 2-minute CPU check with fake data (no download)
set -euo pipefail
MODE=${MODE:-quick}
AMP="--amp"

case "$MODE" in
  smoke) T_EP=1; S_EP=1; NAS_BUDGET=6; NAS_POP=3; NAS_EP=1; BEST_EP=1
         EXTRA="--fake --max-steps 2 --workers 0 --batch-size 32"; NAS_EXTRA="--fake --max-steps 2 --workers 0 --batch-size 32 --latency-runs 10"
         TEACHER_FLAGS="" ;;
  quick) T_EP=30; S_EP=30; NAS_BUDGET=40; NAS_POP=10; NAS_EP=5; BEST_EP=60
         EXTRA=""; NAS_EXTRA=""; TEACHER_FLAGS="--pretrained --lr 0.05" ;;
  full)  T_EP=200; S_EP=200; NAS_BUDGET=100; NAS_POP=20; NAS_EP=10; BEST_EP=200
         EXTRA=""; NAS_EXTRA="--train-subset 45000"; TEACHER_FLAGS="" ;;
esac
echo "== MODE=$MODE =="

# ---------------- Question 1: knowledge distillation ----------------
python -m q1_kd.train_teacher  --epochs $T_EP $TEACHER_FLAGS $AMP $EXTRA
python -m q1_kd.train_baseline --arch all --epochs $S_EP $AMP $EXTRA
python -m q1_kd.train_kd       --arch all --epochs $S_EP --teacher checkpoints/teacher_resnet50.pt \
                               --T 4 --alpha 0.9 --beta 1.0 --feat-loss fitnet $AMP $EXTRA

# ---------------- Question 2: hardware-aware NAS ----------------
python -m q2_nas.search --strategy evolution --budget $NAS_BUDGET --population $NAS_POP \
                        --proxy-epochs $NAS_EP --max-latency-ms 8 --max-size-mb 2 --max-peak-mem-mb 64 $AMP $NAS_EXTRA
python -m q2_nas.train_best --epochs $BEST_EP $AMP ${EXTRA}

# ---------------- Deployment + benchmarking ----------------
python -m deploy.export_onnx --include-teacher
python -m benchmark.benchmark --threads 1 --runs 100
if [ "$MODE" != "smoke" ]; then
  python -m deploy.infer --all --cifar-samples 5
  python -m deploy.onnx_infer --all --cifar-samples 5
fi
python -m tools.make_report_assets
echo "All done. See results/summary.md"
