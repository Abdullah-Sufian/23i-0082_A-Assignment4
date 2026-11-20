"""General helpers: seeding, device selection, checkpoints, CSV logging, model size."""
import csv
import io
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model):
    """Serialized size of the model weights (what you would ship to a device)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def save_checkpoint(path, model, meta):
    """Save only the weights + metadata needed to rebuild the model.

    meta must contain at least: arch, num_classes, dataset, input_size.
    For NAS models it also contains nas_config.
    """
    ensure_dir(os.path.dirname(path) or ".")
    torch.save({"model": model.state_dict(), **meta}, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class CSVLogger:
    """Append rows (dicts) to a CSV file, writing the header on first use."""

    def __init__(self, path, fieldnames=None):
        ensure_dir(os.path.dirname(path) or ".")
        self.path = path
        self.fieldnames = fieldnames
        self._new = True

    def log(self, row):
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        mode = "w" if self._new else "a"
        with open(self.path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if self._new:
                w.writeheader()
                self._new = False
            w.writerow(row)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
