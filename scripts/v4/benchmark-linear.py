#!/usr/bin/env python3
"""Interleaved synchronized FP16/real packed MPS kernels at VGGT matrix sizes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Runtime"))
import torch
from rebuild3d_worker.quantization import PackedLinear, quantize_weight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    if a.output.exists() or a.repeats < 3:
        raise ValueError("New output and at least three repeats required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3021)
    torch.set_num_threads(4)
    report = {"torch": torch.__version__, "fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
              "synchronized": True, "warmupIterations": 2, "repeats": a.repeats, "results": [],
              "scope": "Linear kernel microbenchmark, not full inference or reconstruction"}
    # Seven 518px images -> 7*(37*37 + 5) = 9618 tokens. The model audit
    # must confirm these shapes on real images before using the result to choose.
    with torch.inference_mode():
        for m, k, n in [(7, 1024, 3072), (1374, 1024, 3072), (9618, 1024, 3072),
                        (9618, 1024, 4096), (9618, 4096, 1024)]:
            w = torch.randn((n,k)) * .03
            x = torch.randn((m,k), dtype=torch.float16, device="mps")
            fp16_weight = w.half().to("mps")
            layers = {"fp16": lambda x: torch.nn.functional.linear(x, fp16_weight)}
            for label, bits, group in [("w8", 8, 0), ("w4-g64", 4, 64), ("w4-g128", 4, 128)]:
                q, params = quantize_weight(w, bits, group)
                layers[label] = PackedLinear(q, params, k, n, bits, group)
            times = {label: [] for label in layers}
            for fn in layers.values():
                for _ in range(2):
                    y = fn(x)
                    torch.mps.synchronize()
            labels = list(layers)
            for iteration in range(a.repeats):
                for label in labels[iteration % len(labels):] + labels[:iteration % len(labels)]:
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    y = layers[label](x)
                    torch.mps.synchronize()
                    times[label].append(time.perf_counter()-start)
            ref = layers["fp16"](x).float()
            errors = {}
            for label, fn in layers.items():
                out = fn(x).float()
                errors[label] = {"relativeRMSE": float(((out-ref).square().mean()/ref.square().mean()).sqrt().cpu()),
                                 "finite": bool(torch.isfinite(out).all().cpu())}
            row = {"M": m, "K": k, "N": n, "seconds": times,
                   "medianSeconds": {label: statistics.median(t) for label,t in times.items()}, "error": errors,
                   "audit": {name: layer.audit() for name,layer in layers.items() if isinstance(layer, PackedLinear)}}
            report["results"].append(row)
            a.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ("M", "K", "N", "medianSeconds")}), flush=True)
            del x, w, fp16_weight, layers, ref, out, y, fn
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
