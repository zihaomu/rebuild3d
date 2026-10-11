"""Versioned weight-only MPS kernels. No CPU fallback or full-weight dequantization.

The packed ABI is private to PyTorch; the runtime must pin the tested version.
INT4 disk layout: uint8[N,K/2], even K in the high nibble. Metal repacking
changes layout only. Its affine reconstruction is (q - 8) * scale + offset.
"""
import torch
from torch import nn

TORCH_VERSION = "2.14.1"
FORMAT_VERSION = 1
PACKINGS = {8: "int8-row-symmetric-v1", 4: "uint4-high-even-affine-v1"}


def validate_shape(out_features, in_features, bits, group_size):
    if bits not in PACKINGS:
        raise ValueError("Only INT8 and INT4 weight formats are supported")
    if out_features <= 0 or in_features <= 0 or out_features % 32 or in_features % 32:
        raise ValueError("MPS packed linear dimensions must be positive multiples of 32")
    if bits == 8 and group_size != 0:
        raise ValueError("INT8 uses one scale per output channel, not groups")
    if bits == 4 and (group_size not in (32, 64, 128, 256) or in_features % group_size):
        raise ValueError("Invalid INT4 group size")


def quantize_weight(weight, bits, group_size=0):
    """Offline CPU conversion only; use the exact FP16 weights of the MPS baseline."""
    if weight.device.type != "cpu" or weight.ndim != 2:
        raise ValueError("Expected a CPU Linear weight matrix")
    n, k = weight.shape
    validate_shape(n, k, bits, group_size)
    w = weight.detach().to(torch.float16).float()
    if not torch.isfinite(w).all():
        raise ValueError("Non-finite or FP16-overflowing weights")
    if bits == 8:
        scale = (w.abs().amax(dim=1) / 127).clamp_min(2**-24).half()
        q = (w / scale.float()[:, None]).round().clamp(-127, 127).to(torch.int8)
        return q.contiguous(), scale.contiguous()
    groups = w.reshape(n, k // group_size, group_size)
    low, high = groups.amin(dim=-1), groups.amax(dim=-1)
    scale = ((high - low) / 15).clamp_min(2**-24).half()
    offset = (low + 8 * scale.float()).half()
    q = ((groups - offset.float()[..., None]) / scale.float()[..., None] + 8)
    q = q.round().clamp(0, 15).to(torch.uint8).reshape(n, k)
    packed = ((q[:, ::2] << 4) | q[:, 1::2]).contiguous()
    parameters = torch.stack((scale, offset), dim=-1).transpose(0, 1).contiguous()
    return packed, parameters


def require_mps_kernels():
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise RuntimeError(f"Packed MPS ABI requires PyTorch {TORCH_VERSION}; got {torch.__version__}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Packed VGGT requires an available Apple MPS device")
    for operator in ("_weight_int8pack_mm", "_convert_weight_to_int4pack", "_weight_int4pack_mm"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key("aten::" + operator, "MPS"):
            raise RuntimeError("Missing MPS kernel: " + operator)


class PackedLinear(nn.Module):
    def __init__(self, packed, parameters, in_features, out_features, bits, group_size=0, bias=None):
        super().__init__()
        require_mps_kernels()
        validate_shape(out_features, in_features, bits, group_size)
        expected_shape = (out_features, in_features if bits == 8 else in_features // 2)
        expected_dtype = torch.int8 if bits == 8 else torch.uint8
        param_shape = (out_features,) if bits == 8 else (in_features // group_size, out_features, 2)
        if packed.shape != expected_shape or packed.dtype != expected_dtype:
            raise ValueError("Packed weight shape or dtype mismatch")
        if parameters.shape != param_shape or parameters.dtype != torch.float16:
            raise ValueError("Quantization parameter shape or dtype mismatch")
        # Check on CPU before entering an unchecked private Metal ABI.
        p = parameters.cpu()
        scales = p if bits == 8 else p[..., 0]
        if not torch.isfinite(p).all() or not (scales > 0).all():
            raise ValueError("Invalid quantization scale/offset")
        if bias is not None and (bias.shape != (out_features,) or not torch.isfinite(bias).all()):
            raise ValueError("Invalid bias")
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        weight = packed.to("mps").contiguous()
        if bits == 4:
            weight = torch.ops.aten._convert_weight_to_int4pack(weight, 2)
        self.register_buffer("packed_weight", weight)
        self.register_buffer("quant_parameters", parameters.to("mps").contiguous())
        self.register_buffer("bias", None if bias is None else bias.to("mps", torch.float16).contiguous())
        self.kernel_calls = 0
        self.observed_shapes = set()

    def forward(self, x):
        if x.device.type != "mps" or x.dtype != torch.float16 or x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("PackedLinear requires MPS FP16 activations with matching final dimension")
        flat = x.reshape(-1, self.in_features).contiguous()
        if flat.shape[0] == 0:
            return x.new_empty((*x.shape[:-1], self.out_features))
        self.observed_shapes.add(tuple(flat.shape))
        if self.bits == 8:
            result = torch.ops.aten._weight_int8pack_mm(flat, self.packed_weight, self.quant_parameters)
        else:
            result = torch.ops.aten._weight_int4pack_mm(flat, self.packed_weight, self.group_size, self.quant_parameters)
        self.kernel_calls += 1
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*x.shape[:-1], self.out_features)

    def audit(self):
        return {"bits": self.bits, "groupSize": self.group_size, "packing": PACKINGS[self.bits],
                "kernel": "aten::_weight_int8pack_mm" if self.bits == 8 else "aten::_weight_int4pack_mm",
                "calls": self.kernel_calls, "activationShapes": sorted(self.observed_shapes),
                "weightDevice": str(self.packed_weight.device), "weightDtype": str(self.packed_weight.dtype),
                "weightBytes": self.packed_weight.numel() * self.packed_weight.element_size(),
                "parameterBytes": self.quant_parameters.numel() * self.quant_parameters.element_size(),
                "originalWeightParameters": self.in_features * self.out_features}
