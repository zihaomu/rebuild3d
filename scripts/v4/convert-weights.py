#!/usr/bin/env python3
"""Stream a verified VGGT checkpoint into an immutable, self-described MPS variant.

Development-time operation only. Users never convert weights while generating.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Runtime"))

from safetensors import safe_open
import torch

from rebuild3d_worker.infer import verify_repository
from rebuild3d_worker.model_weights import (METADATA_KEY, WEIGHT_SHA256, digest,
                                          make_recipe, meta_model, verify_weights)
from rebuild3d_worker.quantization import quantize_weight

DTYPES = {torch.float16: ("F16", 2), torch.float32: ("F32", 4), torch.int8: ("I8", 1), torch.uint8: ("U8", 1)}


def transformed(name, tensor, recipe):
    owner = name[:-7] if name.endswith(".weight") else None
    quant = recipe["quantizedModules"].get(owner)
    if quant:
        q, p = quantize_weight(tensor, quant["bits"], quant["groupSize"])
        return [(owner + ".qweight", q), (owner + ".quant_parameters", p)]
    dtype = torch.float16 if name.startswith("aggregator.") else torch.float32
    return [(name, tensor.to(dtype).contiguous())]


def tensor_layout(model, recipe):
    layout = {}
    for name, parameter in model.named_parameters():
        owner = name[:-7] if name.endswith(".weight") else None
        q = recipe["quantizedModules"].get(owner)
        if q:
            n, k = parameter.shape
            layout[owner + ".qweight"] = (torch.int8 if q["bits"] == 8 else torch.uint8, (n, k if q["bits"] == 8 else k//2))
            layout[owner + ".quant_parameters"] = (torch.float16, (n,) if q["bits"] == 8 else (k//q["groupSize"], n, 2))
        else:
            layout[name] = (torch.float16 if name.startswith("aggregator.") else torch.float32, tuple(parameter.shape))
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="New directory, never overwrites earlier candidates")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--policy", choices=["trimmed-fp16", "w8", "w4"], required=True)
    parser.add_argument("--group-size", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    verify_repository(args.repo)
    if digest(args.source) != WEIGHT_SHA256:
        raise ValueError("Original checkpoint SHA-256 mismatch")
    model = meta_model(args.repo)
    recipe = make_recipe(model, args.policy, args.group_size)
    recipe["converterSources"] = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
        ROOT / "Runtime/rebuild3d_worker/model_weights.py", ROOT / "Runtime/rebuild3d_worker/quantization.py"]}
    recipe["weightLicense"] = "CC-BY-NC-4.0; same restriction as original checkpoint"
    layout = tensor_layout(model, recipe)
    header = {"__metadata__": {METADATA_KEY: json.dumps(recipe, sort_keys=True, separators=(",", ":"))}}
    offset = 0
    for name, (dtype, shape) in layout.items():
        label, width = DTYPES[dtype]
        end = offset + math.prod(shape) * width
        header[name] = {"dtype": label, "shape": list(shape), "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "vggt.safetensors"
    partial = destination.with_suffix(".partial")
    with safe_open(args.source, framework="pt", device="cpu") as original, partial.open("xb") as out:
        enabled = set(dict(model.named_parameters()))
        if enabled - set(original.keys()) or any(not key.startswith(("point_head.", "track_head.")) for key in set(original.keys()) - enabled):
            raise ValueError("Original tensor list differs from pinned architecture")
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        data_start = out.tell()
        for index, (name, parameter) in enumerate(model.named_parameters()):
            source = original.get_tensor(name)
            if source.shape != parameter.shape or not torch.isfinite(source).all():
                raise ValueError("Invalid source tensor: " + name)
            for key, tensor in transformed(name, source, recipe):
                if tensor.dtype != layout[key][0] or tuple(tensor.shape) != layout[key][1]:
                    raise ValueError("Conversion layout mismatch: " + key)
                if out.tell() - data_start != header[key]["data_offsets"][0]:
                    raise ValueError("Noncontiguous output write")
                out.write(tensor.numpy().tobytes())
            if index % 200 == 0:
                print("Converted", index, name, flush=True)
        out.flush()
        os.fsync(out.fileno())
    partial.rename(destination)
    # Read every output tensor through the deployed reader, comparing with a
    # second conversion from the immutable source. No full checkpoint in RAM.
    checked = 0
    with safe_open(args.source, framework="pt", device="cpu") as original, safe_open(destination, framework="pt", device="cpu") as candidate:
        if set(candidate.keys()) != set(layout):
            raise ValueError("Written checkpoint tensor list differs")
        for name, _ in model.named_parameters():
            for key, reference in transformed(name, original.get_tensor(name), recipe):
                if not torch.equal(candidate.get_tensor(key), reference):
                    raise ValueError("Written checkpoint differs: " + key)
                checked += 1
    manifest = {"recipe": recipe, "derivedWeightSHA256": digest(destination), "weightBytes": destination.stat().st_size,
                "verifiedTensorCount": checked, "conversionAndVerificationSeconds": time.monotonic() - start}
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_weights(destination)
    print(json.dumps({"output": str(destination), "policy": args.policy, "weightBytes": manifest["weightBytes"],
                      "lowBitParameterFraction": recipe["lowBitParameterFraction"], "sha256": manifest["derivedWeightSHA256"],
                      "seconds": manifest["conversionAndVerificationSeconds"]}), flush=True)


if __name__ == "__main__":
    main()
