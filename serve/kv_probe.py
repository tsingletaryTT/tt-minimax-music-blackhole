# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Run inside the music container's image, on one chip: does a cached tensor over about 32 MiB load back?

On some hosts the image's `ttnn.as_tensor(..., cache_file_name=...)` dumps a large tensor fine and then never returns
from loading it (the thread spins in the kernel). The server hits this when it builds its layer-0 KV cache, so
a boot hangs with no error. This probe asks the same question in a few seconds. It prints `PROBE-OK <seconds>` when
the load returns; the launcher kills it after its timeout when it does not.
"""
import time

import torch
import ttnn

SHAPE = (2, 8, 16384, 128)          # the server's KV cache tensor: 64 MiB in bf16
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    path = "/probe/cache/kv_probe"
    for which in ("dump", "load"):
        t = time.time()
        x = ttnn.as_tensor(torch.zeros(SHAPE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_file_name=path)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(x)
        print(f"{which} {time.time() - t:.2f}s", flush=True)
    print(f"PROBE-OK {time.time() - t:.2f}", flush=True)
finally:
    ttnn.close_mesh_device(dev)
