"""Verified legacy and derived VGGT checkpoints, loaded one tensor at a time."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from safetensors import safe_open
import torch

from .quantization import FORMAT_VERSION, PACKINGS, TORCH_VERSION, PackedLinear, validate_shape

REPO_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
WEIGHT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
WEIGHT_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
METADATA_KEY = "rebuild3d.weight_format"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_path(weights):
    return Path(weights).with_suffix(".json")


def meta_model(repo):
    sys.path.insert(0, str(Path(repo).resolve()))
    from vggt.models.vggt import VGGT
    linspace = torch.linspace
    with patch("torch.linspace", lambda *a, **k: linspace(*a, **{**k, "device": "cpu"})):
        with torch.device("meta"):
            return VGGT(enable_point=False, enable_track=False)


def verify_weights(weights):
    actual = digest(weights)
    if actual == WEIGHT_SHA256:
        return {"format": "vggt-original", "derivedWeightSHA256": actual,
                "baseWeightSHA256": WEIGHT_SHA256, "baseModelRevision": WEIGHT_REVISION,
                "quantizedModules": {}, "backendID": "pytorch-mps-v3"}
    path = manifest_path(weights)
    if not path.is_file():
        raise ValueError("Unknown checkpoint; a verified derived-weight manifest is required")
    manifest = json.loads(path.read_text())
    if manifest.get("derivedWeightSHA256") != actual or manifest.get("weightBytes") != Path(weights).stat().st_size:
        raise ValueError("Derived checkpoint SHA-256/length mismatch")
    recipe = manifest["recipe"]
    if (recipe.get("format") not in ("rebuild3d-vggt", "rebuild3d-vggt-mlx") or recipe.get("formatVersion") != FORMAT_VERSION
            or recipe.get("baseWeightSHA256") != WEIGHT_SHA256
            or recipe.get("baseModelRevision") != WEIGHT_REVISION
            or recipe.get("sourceRevision") != REPO_REVISION):
        raise ValueError("Unsupported derived weight lineage or format")
    if recipe.get("runtimeTorchVersion") != TORCH_VERSION:
        raise ValueError("Unsupported packed-weight ABI")
    if recipe.get("format") == "rebuild3d-vggt-mlx" and recipe.get("runtimeMLXVersion") != "0.32.2":
        raise ValueError("Unsupported MLX packed-weight ABI")
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        embedded = json.loads((checkpoint.metadata() or {}).get(METADATA_KEY, "null"))
        if embedded != recipe:
            raise ValueError("Checkpoint and companion recipe disagree")
    return {**recipe, "derivedWeightSHA256": actual, "manifestSHA256": digest(path),
            "weightBytes": manifest["weightBytes"]}


def make_recipe(model, policy, group_size=0):
    if policy not in ("trimmed-fp16", "w8", "w4"):
        raise ValueError("Unknown conversion policy")
    bits = {"trimmed-fp16": 0, "w8": 8, "w4": 4}[policy]
    if bits != 4 and group_size != 0:
        raise ValueError("Only W4 uses group_size")
    modules = {}
    for name, module in model.named_modules():
        if bits and name.startswith("aggregator.") and isinstance(module, torch.nn.Linear):
            validate_shape(module.out_features, module.in_features, bits, group_size)
            modules[name] = {"bits": bits, "groupSize": group_size, "packing": PACKINGS[bits],
                             "inFeatures": module.in_features, "outFeatures": module.out_features}
    precisions = {name: (f"INT{bits}" if name.endswith(".weight") and name[:-7] in modules else
                        "FP16" if name.startswith("aggregator.") else "FP32")
                  for name, _ in model.named_parameters()}
    total = sum(p.numel() for p in model.parameters())
    low_bit = sum(m["inFeatures"] * m["outFeatures"] for m in modules.values())
    return {"format": "rebuild3d-vggt", "formatVersion": FORMAT_VERSION, "policy": policy,
            "baseWeightSHA256": WEIGHT_SHA256, "baseModelRevision": WEIGHT_REVISION,
            "sourceRevision": REPO_REVISION, "runtimeTorchVersion": TORCH_VERSION,
            "backendID": "pytorch-mps-packed-v1" if bits else "pytorch-mps-trimmed-v1",
            "quantizedModules": modules, "modulePrecisionMap": precisions,
            "enabledParameterCount": total, "lowBitParameterCount": low_bit,
            "lowBitParameterFraction": low_bit / total, "calibrationDatasetHash": None,
            "calibration": "data-free weight-only; no photos used",
            "cpuSupport": "not enabled for derived MPS checkpoint; legacy FP32 checkpoint remains available in development"}


def load_model(repo, weights, device, info=None):
    info = verify_weights(weights) if info is None else info
    if info["format"] == "rebuild3d-vggt-mlx":
        raise ValueError("MLX checkpoint requires the MLXBackend loader; MPS packed layouts are incompatible")
    derived = info["format"] != "vggt-original"
    if derived and device != "mps":
        raise ValueError("Derived checkpoint is MPS-only; no implicit CPU/dequantization fallback")
    model = meta_model(repo)
    parameters = dict(model.named_parameters())
    quantized = info["quantizedModules"]
    expected = set(parameters)
    for name, recipe in quantized.items():
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear) or not name.startswith("aggregator."):
            raise ValueError("Only validated aggregator Linear modules can use this format")
        if (module.in_features != recipe["inFeatures"] or module.out_features != recipe["outFeatures"]
                or recipe["packing"] != PACKINGS.get(recipe["bits"])):
            raise ValueError("Quantized module layout mismatch: " + name)
        expected.remove(name + ".weight")
        expected.update((name + ".qweight", name + ".quant_parameters"))
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        missing, extra = expected - set(checkpoint.keys()), set(checkpoint.keys()) - expected
        if missing or (extra and (derived or not all(k.startswith(("point_head.", "track_head.")) for k in extra))):
            raise ValueError(f"Checkpoint tensor list mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        for name, recipe in quantized.items():
            bias = checkpoint.get_tensor(name + ".bias") if name + ".bias" in expected else None
            module = PackedLinear(checkpoint.get_tensor(name + ".qweight"), checkpoint.get_tensor(name + ".quant_parameters"),
                                  recipe["inFeatures"], recipe["outFeatures"], recipe["bits"], recipe["groupSize"], bias)
            owner, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(owner), leaf, module)
        for name, parameter in model.named_parameters():
            tensor = checkpoint.get_tensor(name)
            dtype = torch.float16 if name.startswith("aggregator.") and device == "mps" else torch.float32
            if tensor.shape != parameter.shape or (derived and tensor.dtype != dtype):
                raise ValueError("Checkpoint tensor shape/precision mismatch: " + name)
            owner, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(owner), leaf, torch.nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False))
    dtype = torch.float16 if device == "mps" else torch.float32
    for name, values in (("_resnet_mean", [.485, .456, .406]), ("_resnet_std", [.229, .224, .225])):
        setattr(model.aggregator, name, torch.tensor(values, device=device, dtype=dtype).view(1, 1, 3, 1, 1))
    if any(p.is_meta for p in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Uninitialized model tensor")
    return model.eval(), len(extra)


def kernel_audit(model):
    return {name: module.audit() for name, module in model.named_modules() if isinstance(module, PackedLinear)}
