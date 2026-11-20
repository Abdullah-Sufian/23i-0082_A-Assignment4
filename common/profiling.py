"""
Hardware profiling helpers (CPU latency, storage size, peak inference memory).

Latency is measured with real wall-clock timing on the CPU (batch size 1 by
default, i.e. the typical edge-device scenario), after warm-up runs.

Peak inference memory is measured as the increase in process RSS (resident set
size) observed by a background sampling thread while inference runs. To get a
clean, comparable number it should be called in a *fresh* process (see
`profile_in_subprocess`), because the allocator keeps memory from earlier work.
"""
import gc
import io
import os
import statistics
import threading
import time

import numpy as np
import torch

try:
    import psutil
except ImportError:  # peak memory becomes "unsupported"
    psutil = None


def _stats(times_ms):
    times_ms = sorted(times_ms)
    return {
        "latency_mean_ms": statistics.mean(times_ms),
        "latency_p50_ms": times_ms[len(times_ms) // 2],
        "latency_p95_ms": times_ms[min(len(times_ms) - 1, int(0.95 * len(times_ms)))],
        "latency_std_ms": statistics.pstdev(times_ms),
    }


class PeakRSSMonitor:
    """Samples the process RSS in a background thread and remembers the maximum."""

    def __init__(self, interval_s=0.0005):
        self.interval = interval_s
        self.supported = psutil is not None
        self.peak = 0
        self.baseline = 0

    def __enter__(self):
        if not self.supported:
            return self
        gc.collect()
        self.proc = psutil.Process(os.getpid())
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *a):
        if self.supported:
            self._stop.set()
            self._t.join()
            self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_delta_mb(self):
        if not self.supported:
            return float("nan")
        return (self.peak - self.baseline) / (1024 ** 2)


@torch.no_grad()
def torch_cpu_latency(model, input_size=32, batch_size=1, warmup=10, runs=50, threads=None):
    """Measure PyTorch CPU latency (ms per batch) and peak RSS increase."""
    if threads:
        torch.set_num_threads(threads)
    model = model.eval().to("cpu")
    x = torch.randn(batch_size, 3, input_size, input_size)
    with PeakRSSMonitor() as mon:
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    out = _stats(times)
    out["peak_mem_mb"] = mon.peak_delta_mb
    out["throughput_img_s"] = 1000.0 * batch_size / out["latency_mean_ms"]
    return out


def onnx_cpu_latency(onnx_path, input_size=32, batch_size=1, warmup=10, runs=50, threads=None):
    """Measure ONNX Runtime CPUExecutionProvider latency and peak RSS increase."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    with PeakRSSMonitor() as mon:
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        x = np.random.randn(batch_size, 3, input_size, input_size).astype(np.float32)
        for _ in range(warmup):
            sess.run(None, {name: x})
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, {name: x})
            times.append((time.perf_counter() - t0) * 1000)
    out = _stats(times)
    out["peak_mem_mb"] = mon.peak_delta_mb
    out["throughput_img_s"] = 1000.0 * batch_size / out["latency_mean_ms"]
    return out


def model_storage_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


@torch.no_grad()
def estimate_activation_mb(model, input_size=32):
    """Analytical fallback: largest (input + output) activation of any leaf module, fp32."""
    peak = [0]

    def hook(m, inp, out):
        b = sum(t.numel() for t in inp if torch.is_tensor(t)) + (out.numel() if torch.is_tensor(out) else 0)
        peak[0] = max(peak[0], b * 4)

    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    model.eval()(torch.randn(1, 3, input_size, input_size))
    for h in hs:
        h.remove()
    return peak[0] / (1024 ** 2)
