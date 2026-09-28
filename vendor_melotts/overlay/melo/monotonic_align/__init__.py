import os
import time

from numpy import zeros, int32, float32
from torch import from_numpy

from .core import maximum_path_jit

_CALL = 0


def maximum_path(neg_cent, mask):
    global _CALL
    _CALL += 1

    profile = os.environ.get("MELOTTS_PROFILE_MAS", "0") == "1"

    if profile:
        t0 = time.perf_counter()

    device = neg_cent.device
    dtype = neg_cent.dtype

    neg_cent = neg_cent.data.cpu().numpy().astype(float32)

    if profile:
        t1 = time.perf_counter()

    path = zeros(neg_cent.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)

    if profile:
        t2 = time.perf_counter()

    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)

    if profile:
        t3 = time.perf_counter()

    result = from_numpy(path).to(device=device, dtype=dtype)

    if profile:
        # Force completion of H2D transfer only for profiling.
        if device.type == "cuda":
            import torch
            torch.cuda.synchronize()

        t4 = time.perf_counter()

        print(
            f"[MAS {_CALL}] "
            f"shape={tuple(neg_cent.shape)} "
            f"gpu_to_cpu={t1-t0:.3f}s "
            f"prep={t2-t1:.3f}s "
            f"search={t3-t2:.3f}s "
            f"cpu_to_gpu={t4-t3:.3f}s "
            f"total={t4-t0:.3f}s",
            flush=True,
        )

    return result
