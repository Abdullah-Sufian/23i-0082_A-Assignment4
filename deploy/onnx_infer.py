"""
CPU inference with ONNX Runtime only (no PyTorch model code needed at run time
except for optional CIFAR loading) - this is what would run on the edge device.

Examples
  python -m deploy.onnx_infer --model onnx/kd_mobilenetv2.onnx --images samples/*.png
  python -m deploy.onnx_infer --model onnx/nas_best.onnx --eval            # test accuracy with ORT
  python -m deploy.onnx_infer --all --benchmark                            # latency of every ONNX model
"""
import argparse
import glob
import os

from common.profiling import onnx_cpu_latency
from deploy.infer import run as run_infer
from deploy.pipeline import InferencePipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="path to .onnx (metadata .json must sit next to it)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--onnx-dir", default="onnx")
    p.add_argument("--images", nargs="*")
    p.add_argument("--folder")
    p.add_argument("--cifar-samples", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="measure CPU latency (batch 1)")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx"))) if args.all else [args.model]
    for path in paths:
        pipe = InferencePipeline(path, backend="onnx", threads=args.threads)
        if args.benchmark:
            r = onnx_cpu_latency(path, pipe.input_size, 1, threads=args.threads)
            print(f"{pipe.name:32s} ORT-CPU latency {r['latency_mean_ms']:.3f} ms (p95 {r['latency_p95_ms']:.3f}) "
                  f"| {r['throughput_img_s']:.0f} img/s")
            continue
        if not (args.images or args.folder or args.cifar_samples or args.eval):
            args.cifar_samples = 5
        run_infer(pipe, args)


if __name__ == "__main__":
    main()
