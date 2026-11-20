"""
Hardware-aware profiler + deployment-constraint checker for NAS candidates.

For every candidate architecture we measure, on the actual CPU:
  * inference latency  (batch 1, after warm-up; mean / p50 / p95 in ms)
  * model storage size (serialized fp32 weights in MB)
  * peak inference memory (RSS increase during inference, MB) where psutil is available
  * parameter count

Measurements run in a *fresh spawned process* by default, so (a) training on the
GPU / other threads does not disturb the timing and (b) the memory allocator has
no memory cached from previous candidates, which would otherwise hide the true
peak. Untrained weights are fine for this: latency, size and memory do not
depend on weight values.
"""
import multiprocessing as mp
from dataclasses import asdict, dataclass

import torch


@dataclass
class Constraints:
    max_latency_ms: float = 8.0      # CPU latency budget per image (batch 1)
    max_size_mb: float = 2.0         # storage budget (fp32 weights)
    max_peak_mem_mb: float = 64.0    # peak inference memory budget
    max_params_m: float = 1e9        # optional parameter budget (millions)

    def check(self, prof):
        """Return (is_feasible, list_of_violation_reasons)."""
        reasons = []
        if prof["latency_mean_ms"] > self.max_latency_ms:
            reasons.append(f"latency {prof['latency_mean_ms']:.2f}ms > {self.max_latency_ms}ms")
        if prof["size_mb"] > self.max_size_mb:
            reasons.append(f"size {prof['size_mb']:.2f}MB > {self.max_size_mb}MB")
        pm = prof.get("peak_mem_mb")
        if pm is not None and pm == pm and pm > self.max_peak_mem_mb:  # pm == pm filters NaN (unsupported)
            reasons.append(f"peak_mem {pm:.1f}MB > {self.max_peak_mem_mb}MB")
        if prof["params_m"] > self.max_params_m:
            reasons.append(f"params {prof['params_m']:.2f}M > {self.max_params_m}M")
        return len(reasons) == 0, reasons

    def as_dict(self):
        return asdict(self)


def _profile(config, num_classes, threads, warmup, runs):
    from common.profiling import estimate_activation_mb, model_storage_mb, torch_cpu_latency
    from common.utils import count_params
    from q2_nas.generator import build_from_config
    torch.manual_seed(0)
    if threads:
        torch.set_num_threads(threads)
    model = build_from_config(config, num_classes).eval()
    res = config["resolution"]
    lat = torch_cpu_latency(model, res, 1, warmup, runs, threads)
    return {
        "params_m": count_params(model) / 1e6,
        "size_mb": model_storage_mb(model),
        "act_mem_est_mb": estimate_activation_mb(model, res),
        **lat,
    }


def _child(q, *args):
    try:
        q.put(("ok", _profile(*args)))
    except Exception as e:  # report failure instead of hanging
        q.put(("err", repr(e)))


def profile_config(config, num_classes=10, threads=1, warmup=10, runs=50, isolated=True, timeout=300):
    """Profile one architecture on the CPU. isolated=True runs it in a fresh process."""
    if not isolated:
        return _profile(config, num_classes, threads, warmup, runs)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, config, num_classes, threads, warmup, runs))
    p.start()
    status, out = q.get(timeout=timeout)
    p.join()
    if status != "ok":
        raise RuntimeError(f"profiling failed: {out}")
    return out
