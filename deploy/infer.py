"""
Unified inference CLI for all trained models (students, distilled students, NAS model).

Input sources (pick one):
  --images a.png b.jpg      your own images
  --folder some_dir/        every image in a folder
  --cifar-samples N         N random images from the CIFAR test set (shows ground truth)
  --eval                    full test-set accuracy

Examples
  python -m deploy.infer --model checkpoints/kd_mobilenetv2.pt --cifar-samples 8
  python -m deploy.infer --model onnx/kd_squeezenet.onnx --images samples/cat.png      # ONNX Runtime
  python -m deploy.infer --all --cifar-samples 4                                       # every model
  python -m deploy.infer --model checkpoints/nas_best.pt --eval
"""
import argparse
import glob
import os
import random

import numpy as np

from deploy.pipeline import InferencePipeline

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def cifar_samples(dataset, root, n, seed):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root=root, train=False, download=True)
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [np.array(ds[i][0]) for i in idx], [ds.classes[ds[i][1]] for i in idx]


def run(pipe, args):
    print(f"\n=== {pipe.name}  [{pipe.backend}]  arch={pipe.meta.get('arch')}  dataset={pipe.dataset} "
          f"input={pipe.input_size}px ===")
    if args.eval:
        from common.data import get_loaders
        _, _, test = get_loaders(pipe.dataset, args.data, 256, args.workers, fake=args.fake)
        acc, n, sec = pipe.evaluate_loader(test)
        print(f"Test accuracy: {acc:.2f}% on {n} images ({1000 * sec / max(1, n):.3f} ms/img)")
        return
    if args.cifar_samples:
        images, truth = cifar_samples(pipe.dataset, args.data, args.cifar_samples, args.seed)
        labels = [os.path.basename(f"cifar_test_{i}") for i in range(len(images))]
    else:
        files = args.images or sorted(f for f in glob.glob(os.path.join(args.folder, "*")) if f.lower().endswith(IMG_EXT))
        images, truth, labels = files, [None] * len(files), [os.path.basename(f) for f in files]
    preds, ms = pipe.predict(images, topk=args.topk)
    for name, gt, p in zip(labels, truth, preds):
        top = ", ".join(f"{c} {100 * s:.1f}%" for c, s in p)
        mark = "" if gt is None else (f"  (truth: {gt}) " + ("OK" if p[0][0] == gt else "WRONG"))
        print(f"{name:20s} -> {top}{mark}")
    print(f"batch of {len(images)} processed in {ms:.1f} ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help=".pt checkpoint or .onnx file")
    p.add_argument("--all", action="store_true", help="run every model in --ckpt-dir (or --onnx-dir with --backend onnx)")
    p.add_argument("--backend", choices=["torch", "onnx"], default=None)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--onnx-dir", default="onnx")
    p.add_argument("--images", nargs="*")
    p.add_argument("--folder")
    p.add_argument("--cifar-samples", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()

    if args.all:
        if args.backend == "onnx":
            paths = sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))
        else:
            paths = sorted(c for c in glob.glob(os.path.join(args.ckpt_dir, "*.pt")) if "teacher" not in c)
    else:
        assert args.model, "give --model or --all"
        paths = [args.model]
    if not (args.images or args.folder or args.cifar_samples or args.eval):
        args.cifar_samples = 5
    for path in paths:
        run(InferencePipeline(path, backend=args.backend, threads=args.threads), args)


if __name__ == "__main__":
    main()
