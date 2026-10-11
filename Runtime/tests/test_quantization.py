import os
import unittest

import numpy as np
import torch

from rebuild3d_worker.quantization import PackedLinear, quantize_weight, validate_shape


class ConversionTests(unittest.TestCase):
    def test_nonfinite_and_invalid_shapes_are_rejected_before_native_kernel(self):
        for bits, group, shape in [(8, 64, (32, 128)), (4, 48, (32, 128)), (4, 256, (32, 128)), (8, 0, (31, 128))]:
            with self.subTest(bits=bits, group=group, shape=shape), self.assertRaises(ValueError):
                validate_shape(*shape, bits, group)
        for value in [float('nan'), float('inf'), 70000.]:
            with self.assertRaises(ValueError):
                quantize_weight(torch.full((32, 128), value), 8)

    def test_signed_and_constant_weights_have_finite_compressed_representation(self):
        w = torch.linspace(-.1, .1, 32 * 128).reshape(32, 128)
        w[0] = 0; w[1] = .25; w[2] = -.25
        for bits, group in [(8, 0), (4, 32), (4, 64), (4, 128)]:
            q, parameters = quantize_weight(w, bits, group)
            self.assertTrue(torch.isfinite(parameters).all())
            if bits == 8:
                restored = q.float() * parameters.float()[:, None]
            else:
                codes = torch.stack([q >> 4, q & 15], dim=-1).reshape(32, 128).float()
                scale, offset = parameters.permute(1, 0, 2).float().unbind(-1)
                restored = ((codes.reshape(32, -1, group) - 8) * scale[..., None] + offset[..., None]).reshape_as(w)
            self.assertLess((restored - w.half().float()).abs().max().item(), .002)
            self.assertLess(q.numel()*q.element_size() + parameters.numel()*parameters.element_size(), w.numel()*2)


@unittest.skipUnless(torch.backends.mps.is_available(), 'Requires real Apple MPS GPU')
class MPSKernelTests(unittest.TestCase):
    def test_int4_nibble_order_affine_offset_groups_bias_and_noncontiguous_input(self):
        self.assertEqual(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), '0', 'Run kernel tests with fallback disabled')
        n, k = 32, 256
        codes = (np.arange(n*k).reshape(n, k) * 7 + np.arange(n)[:, None]) % 16
        packed = ((codes[:, ::2] << 4) | codes[:, 1::2]).astype('uint8')
        for group in [32, 64, 128, 256]:
            s = np.linspace(.005, .035, n*k//group).reshape(n, k//group).astype('float16')
            z = np.linspace(-.2, .3, n*k//group).reshape(n, k//group).astype('float16')
            params = np.stack([s, z], -1).transpose(1, 0, 2).copy()
            bias = np.linspace(-.1, .1, n).astype('float16')
            layer = PackedLinear(torch.from_numpy(packed), torch.from_numpy(params), k, n, 4, group, torch.from_numpy(bias))
            # Independent NumPy reference, including different group/channel offsets.
            weight = (codes.reshape(n, -1, group).astype('float32')-8)*s.astype('float32')[..., None] + z.astype('float32')[..., None]
            for rows in [1, 7, 13, 33]:
                x = np.sin(np.arange(rows*k).reshape(1, rows, k)*.03).astype('float16')
                noncontiguous = torch.from_numpy(np.repeat(x, 2, axis=-1)).to('mps')[..., ::2]
                self.assertFalse(noncontiguous.is_contiguous())
                result = layer(noncontiguous).cpu().float().numpy().reshape(rows, n)
                reference = (x.reshape(rows, k).astype('float32') @ weight.reshape(n, k).T).astype('float16') + bias
                np.testing.assert_allclose(result, reference.astype('float32'), rtol=.004, atol=.008)
            self.assertEqual(layer.packed_weight.dtype, torch.int32)
            self.assertEqual(layer.audit()['calls'], 4)

    def test_int8_small_and_large_row_kernels_match_reference(self):
        n, k = 64, 128
        q = (np.arange(n*k).reshape(n,k) % 255 - 127).astype('int8')
        scale = np.linspace(.001, .02, n).astype('float16')
        layer = PackedLinear(torch.from_numpy(q), torch.from_numpy(scale), k, n, 8)
        for rows in [1, 7, 11, 12, 33]:
            x = np.cos(np.arange(rows*k).reshape(rows,k)*.1).astype('float16')
            reference = x.astype('float32') @ (q.astype('float32')*scale.astype('float32')[:, None]).T
            result = layer(torch.from_numpy(x).to('mps')).cpu().float().numpy()
            np.testing.assert_allclose(result, reference, rtol=.005, atol=.01)
        self.assertEqual(layer.packed_weight.dtype, torch.int8)
        with self.assertRaises(ValueError):
            layer(torch.ones((1,k)))

    def test_invalid_scales_are_rejected(self):
        for value in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                PackedLinear(torch.zeros((32,128), dtype=torch.int8), torch.full((32,), value, dtype=torch.float16), 128, 32, 8)


if __name__ == '__main__':
    unittest.main()
