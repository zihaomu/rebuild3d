"""Managed prepacked MLX backbone and manifest-precision MPS output heads.

This module is imported only for MLX artifacts. Floating legacy runtimes do not
need an MLX installation. All model computation stays on the GPU; resizing the
small learned position grid is explicit, static checkpoint preprocessing.
"""
import importlib.metadata
import os
import platform

import mlx.core as mx
import numpy as np
from safetensors import safe_open
import torch

from .mlx_aggregator import Aggregator, ROTARY_POSITION_POLICY, EXECUTION_POLICY
from .model_weights import meta_model

BACKEND_ID = "mlx-aggregator-pytorch-heads-v1"
HEADS16 = "fp16-parameters-fp32-position-and-activation-v1"


class MLXBackend:
    def __init__(self, repo, weights, info, size, device):
        if device != "mps" or platform.machine() != "arm64" or not torch.backends.mps.is_available():
            raise ValueError("MLX sparse backend requires Apple Silicon GPU; no CPU fallback")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise ValueError("MLX backend requires PYTORCH_ENABLE_MPS_FALLBACK=0 before Python starts")
        if info.get("backendID") != BACKEND_ID or info.get("format") != "rebuild3d-vggt-mlx":
            raise ValueError("Unsupported MLX backend or checkpoint format")
        self.versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal")}
        if any(version != info["runtimeMLXVersion"] for version in self.versions.values()):
            raise ValueError("MLX/Metal runtime version differs from the checkpoint ABI")
        if torch.__version__.split("+")[0] != info["runtimeTorchVersion"]:
            raise ValueError("PyTorch runtime version differs from the checkpoint ABI")
        precision = info.get("headPrecision", "fp32")
        if precision not in ("fp32", HEADS16):
            raise ValueError("Unsupported output-head precision")
        self.head_dtype = torch.float16 if precision == HEADS16 else torch.float32
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(256 * 1024**2)
        self.model = meta_model(repo)
        self.aggregator = Aggregator(weights, self.model.aggregator, weight_info=info, size=size)
        self.model.aggregator = torch.nn.Identity()
        with safe_open(weights, framework="pt", device="cpu") as checkpoint:
            expected = dict(self.model.named_parameters())
            if {k for k in checkpoint.keys() if not k.startswith("aggregator.")} != set(expected):
                raise ValueError("MLX output-head tensor set mismatch")
            for name, parameter in expected.items():
                value = checkpoint.get_tensor(name)
                if value.shape != parameter.shape or value.dtype != self.head_dtype:
                    raise ValueError("MLX output-head shape/precision mismatch: " + name)
                owner, leaf = name.rsplit(".", 1)
                setattr(self.model.get_submodule(owner), leaf,
                        torch.nn.Parameter(value.to("mps"), requires_grad=False))
        if any(p.is_meta for p in list(self.model.parameters()) + list(self.model.buffers())):
            raise ValueError("Uninitialized MLX output-head tensor")
        self.model.eval()
        if self.head_dtype == torch.float16:
            apply_position = self.model.depth_head._apply_pos_embed
            self.model.depth_head._apply_pos_embed = (
                lambda x, *a, **k: apply_position(x.float(), *a, **k).to(x.dtype))
        self.config = {"backendID": BACKEND_ID, "backendVersion": 4,
                       "rotaryPositionPolicy": ROTARY_POSITION_POLICY,
                       "executionPolicy": EXECUTION_POLICY,
                       "runtimeVersions": self.versions, "headPrecision": precision,
                       "modelImageSize": size, "weightPacking": "mlx-affine-uint32-v1",
                       "groupSize": self.aggregator.group_size, "prepacked": True,
                       "linearComputePolicy": self.aggregator.linear_policy,
                       "calibrationDatasetHash": info.get("calibrationDatasetHash")}

    @staticmethod
    def synchronize():
        mx.synchronize()

    @staticmethod
    def memory_stats():
        return {"mlxActiveBytes": mx.get_active_memory(), "mlxCacheBytes": mx.get_cache_memory(),
                "mlxPeakActiveBytes": mx.get_peak_memory()}

    def predict(self, images, timed, progress):
        # Scaling is performed by the common Torch input path, exactly as v3.
        with timed("frameworkInputTransfer", gpu=True):
            mlx_images = mx.array(images.cpu().numpy())
            mx.eval(mlx_images)
        with timed("aggregator", gpu=True):
            tokens, patch_start = self.aggregator(mlx_images, progress)
        with timed("frameworkTransferAndRelease", gpu=True):
            tokens = [torch.from_numpy(np.asarray(t).copy()).to("mps", self.head_dtype)
                      if t is not None else None for t in tokens]
            audit = self.aggregator.audit()
            self.aggregator = None
            del mlx_images
            mx.clear_cache()
        with torch.inference_mode():
            with timed("cameraHeadAndReadback", gpu=True):
                pose = self.model.camera_head(tokens)[-1].float().cpu()
            with timed("depthHeadAndReadback", gpu=True):
                # Explicitly retain the pinned activation in FP32. This worker
                # executes one inference only; restore the module even on error.
                from vggt.heads import dpt_head
                activate = dpt_head.activate_head
                try:
                    dpt_head.activate_head = lambda out, **kw: activate(out.float(), **kw)
                    depth, confidence = self.model.depth_head(
                        tokens, images.to(self.head_dtype), patch_start, frames_chunk_size=1)
                finally:
                    dpt_head.activate_head = activate
                depth = depth.float().cpu().numpy()[0, ..., 0]
                confidence = confidence.float().cpu().numpy()[0]
        calls = [call for block in audit["blocks"].values() for call in block["dispatches"].values()]
        audit["packedKernelCalls"] = sum(call["calls"] for call in calls)
        audit["executedLowBitParameterCount"] = sum(call["originalWeightParameters"] for call in calls if call["calls"])
        return pose, depth, confidence, audit
