#!/usr/bin/env python3
"""Benchmark YOLO26 fairly across Torch, Ultralytics, and KernRT.

The three model-only measurements use the same fixed 1x3x640x640 f32 input,
ten warmups, one CPU thread, and the same iteration count. Ultralytics' high
level ``predict`` path is reported separately because it includes image
preprocessing and result construction rather than just the neural graph.

Usage:
  python scripts/benchmark_yolo26.py --weights yolo26n.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def bench(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-ultralytics-e2e", action="store_true")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    # Apple Accelerate/vecLib can otherwise create its own worker pool. Keep
    # the comparison and KernRT's single-thread contract genuinely one-thread.
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)

    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Install benchmark dependencies first: pip install ultralytics torch numpy"
        ) from exc

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = YOLO(args.weights)
    onnx_path = args.onnx
    if onnx_path is None:
        exported = model.export(
            format="onnx",
            imgsz=args.imgsz,
            nms=False,
            dynamic=False,
            simplify=True,
            half=False,
        )
        onnx_path = Path(str(exported))
    if not onnx_path.exists():
        raise SystemExit(f"ONNX export not found: {onnx_path}")

    # A deterministic input makes output checks and repeated runs comparable.
    rng = np.random.default_rng(0x26)
    nchw = rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    torch_input = torch.from_numpy(nchw)
    torch_model = model.model.fuse().eval()

    with torch.inference_mode():
        torch_seconds = bench(
            lambda: torch_model(torch_input), 10, args.iterations
        )

    ultralytics_seconds = None
    if not args.skip_ultralytics_e2e:
        # High-level inference is useful to users, but it includes preprocessing
        # and Results allocation, so it must not be mixed with model-only rates.
        image = (nchw[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        ultralytics_seconds = bench(
            lambda: model.predict(
                image,
                imgsz=args.imgsz,
                device="cpu",
                nms=False,
                verbose=False,
            ),
            10,
            args.iterations,
        )

    binary = Path("target/release/examples/yolo26_server")
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "--release", "--example", "yolo26_server"],
            check=True,
        )
    rust_command = [str(binary), str(onnx_path), "--bench", str(args.iterations)]
    rust = subprocess.run(
        rust_command,
        check=True,
        text=True,
        capture_output=True,
    )

    print(f"model={args.weights} onnx={onnx_path} threads={args.threads}")
    print(f"torch raw:        {1 / torch_seconds:8.1f} img/s ({torch_seconds * 1e3:.2f} ms)")
    if ultralytics_seconds is not None:
        print(
            f"ultralytics e2e:  {1 / ultralytics_seconds:8.1f} img/s "
            f"({ultralytics_seconds * 1e3:.2f} ms)"
        )
    backend_line = next(line for line in rust.stdout.splitlines() if line.startswith("kernrt "))
    print(f"kernrt cpu:       {backend_line.removeprefix('kernrt yolo26: ')}")


if __name__ == "__main__":
    main()
