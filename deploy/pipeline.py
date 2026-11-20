"""
Unified inference pipeline for every trained model (3 baseline students,
3 distilled students, the NAS model, and optionally the teacher).

    pipe = InferencePipeline("checkpoints/kd_mobilenetv2.pt")              # PyTorch backend
    pipe = InferencePipeline("onnx/kd_mobilenetv2.onnx", backend="onnx")   # ONNX Runtime (CPU)
    preds = pipe.predict(["cat.png", "ship.jpg"], topk=5)

Both backends share the same preprocessing (resize -> tensor -> normalise with
the training-set statistics), so the outputs are directly comparable. All the
metadata needed for preprocessing (dataset, input size, class names) comes from
the checkpoint or from the JSON file written next to the .onnx model.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.classes import class_names
from common.data import STATS


def preprocess(images, dataset, size):
    """List of PIL images / paths / HxWx3 uint8 arrays -> normalised float32 NCHW numpy array."""
    mean = np.array(STATS[dataset][0], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(STATS[dataset][1], dtype=np.float32).reshape(1, 1, 3)
    batch = []
    for im in images:
        if isinstance(im, str):
            im = Image.open(im)
        elif isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
        batch.append(arr.transpose(2, 0, 1))
    return np.stack(batch).astype(np.float32)


class InferencePipeline:
    def __init__(self, model_path, backend=None, threads=None, device="cpu"):
        self.backend = backend or ("onnx" if model_path.endswith(".onnx") else "torch")
        self.path = model_path
        if self.backend == "torch":
            from common.utils import load_checkpoint
            from models.registry import model_from_checkpoint
            ckpt = load_checkpoint(model_path)
            self.meta = {k: v for k, v in ckpt.items() if k != "model"}
            if threads:
                torch.set_num_threads(threads)
            self.device = torch.device(device)
            self.model = model_from_checkpoint(ckpt).to(self.device).eval()
        else:
            import onnxruntime as ort
            meta_path = os.path.splitext(model_path)[0] + ".json"
            with open(meta_path) as f:
                self.meta = json.load(f)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if threads:
                so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        self.dataset = self.meta["dataset"]
        self.input_size = self.meta["input_size"]
        self.classes = class_names(self.dataset)
        self.name = self.meta.get("run", os.path.basename(model_path))

    def logits(self, x):
        """x: float32 NCHW numpy array, already normalised (and at any resolution)."""
        if x.shape[-1] != self.input_size:
            x = F.interpolate(torch.from_numpy(x), size=(self.input_size,) * 2, mode="bilinear",
                              align_corners=False).numpy()
        if self.backend == "torch":
            with torch.no_grad():
                return self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]

    def predict(self, images, topk=5):
        t0 = time.perf_counter()
        x = preprocess(images, self.dataset, self.input_size)
        z = self.logits(x)
        ms = (time.perf_counter() - t0) * 1000
        z = z - z.max(1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        out = []
        for p in probs:
            idx = p.argsort()[::-1][:topk]
            out.append([(self.classes[i], float(p[i])) for i in idx])
        return out, ms

    def evaluate_loader(self, loader, max_batches=None):
        """Top-1 accuracy over a DataLoader of already-normalised tensors."""
        correct = n = 0
        t0 = time.perf_counter()
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            pred = self.logits(x.numpy()).argmax(1)
            correct += int((pred == np.asarray(y)).sum())
            n += len(y)
        return 100.0 * correct / max(1, n), n, time.perf_counter() - t0
