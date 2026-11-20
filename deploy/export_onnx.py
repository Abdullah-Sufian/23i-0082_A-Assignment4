"""
Export trained checkpoints to ONNX and verify them against PyTorch.

For each checkpoint this writes
    onnx/<run>.onnx   the model (dynamic batch dimension, opset 17)
    onnx/<run>.json   preprocessing metadata (dataset, input size, normalisation, classes)
and checks with onnx.checker + a numerical comparison of ONNX Runtime vs PyTorch outputs.

Examples
  python -m deploy.export_onnx                         # all students + NAS model in checkpoints/
  python -m deploy.export_onnx --include-teacher
  python -m deploy.export_onnx --ckpts checkpoints/kd_mobilenetv2.pt
"""
import argparse
import glob
import os
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

from common.classes import class_names
from common.data import STATS
from common.utils import ensure_dir, file_size_mb, load_checkpoint, save_json
from models.registry import model_from_checkpoint


def export_one(ckpt_path, out_dir, opset=17):
    ckpt = load_checkpoint(ckpt_path)
    model = model_from_checkpoint(ckpt).cpu().eval()
    run = ckpt.get("run", os.path.splitext(os.path.basename(ckpt_path))[0])
    size = ckpt["input_size"]
    x = torch.randn(1, 3, size, size)
    onnx_path = os.path.join(out_dir, f"{run}.onnx")
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, do_constant_folding=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tracer warnings about the return_features branch are harmless
        try:
            torch.onnx.export(model, (x,), onnx_path, dynamo=False, **kwargs)   # TorchScript-based exporter
        except TypeError:                                                      # older torch: no `dynamo` arg
            torch.onnx.export(model, (x,), onnx_path, **kwargs)

    onnx.checker.check_model(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    xb = torch.randn(4, 3, size, size)
    with torch.no_grad():
        ref = model(xb).numpy()
    got = sess.run(None, {"input": xb.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    rel_diff = max_diff / max(1e-12, float(np.abs(ref).max()))
    same_pred = bool((ref.argmax(1) == got.argmax(1)).all())

    ds = ckpt["dataset"]
    meta = {"run": run, "arch": ckpt["arch"], "method": ckpt.get("method"), "dataset": ds,
            "num_classes": ckpt["num_classes"], "input_size": size,
            "mean": STATS[ds][0], "std": STATS[ds][1], "classes": class_names(ds),
            "pytorch_test_acc": ckpt.get("acc"), "source_checkpoint": ckpt_path, "opset": opset}
    save_json(meta, os.path.join(out_dir, f"{run}.json"))
    print(f"{run:32s} -> {onnx_path} ({file_size_mb(onnx_path):.2f} MB) max|torch-ort|={max_diff:.2e} (rel {rel_diff:.1e}) "
          f"same_top1={same_pred}")
    return {"run": run, "arch": ckpt["arch"], "onnx_path": onnx_path, "onnx_size_mb": file_size_mb(onnx_path),
            "max_abs_diff": max_diff, "max_rel_diff": rel_diff, "same_top1": same_pred}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--ckpts", nargs="*", help="explicit checkpoint list (default: all in --ckpt-dir)")
    p.add_argument("--out", default="onnx")
    p.add_argument("--include-teacher", action="store_true")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--results", default="results/onnx_export.csv")
    args = p.parse_args()

    ensure_dir(args.out)
    ckpts = args.ckpts or sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    if not args.include_teacher and not args.ckpts:
        ckpts = [c for c in ckpts if "teacher" not in os.path.basename(c)]
    rows = [export_one(c, args.out, args.opset) for c in ckpts]
    ensure_dir(os.path.dirname(args.results) or ".")
    pd.DataFrame(rows).to_csv(args.results, index=False)
    print(f"Exported {len(rows)} models -> {args.out}/ (summary: {args.results})")


if __name__ == "__main__":
    main()
