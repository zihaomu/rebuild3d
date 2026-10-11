"""Managed MLX packing must execute directly and fail closed on invalid inputs."""
import importlib.util
import unittest

import numpy as np
import torch

HAS_MLX = importlib.util.find_spec("mlx") is not None


@unittest.skipUnless(HAS_MLX and torch.backends.mps.is_available(), "Requires MLX and an accessible Apple GPU")
class MLXRuntimeTests(unittest.TestCase):
    def test_native_rotary_grid_and_special_tokens_are_unchanged(self):
        import mlx.core as mx
        from rebuild3d_worker.mlx_aggregator import rotary_positions
        positions = rotary_positions(3, 518)
        self.assertEqual(positions.dtype, mx.int32)
        actual = np.asarray(positions)
        expected = np.array([(row+1, col+1) for row in range(37) for col in range(37)])
        for view in range(3):
            np.testing.assert_array_equal(actual[view, :5], np.zeros((5, 2)))
            np.testing.assert_array_equal(actual[view, 5:], expected)

    def test_lower_resolution_preserves_image_relative_patch_centers(self):
        from rebuild3d_worker.mlx_aggregator import rotary_positions
        actual = np.asarray(rotary_positions(12, 392))
        expected = np.array([((row+.5)/28, (col+.5)/28)
                             for row in range(28) for col in range(28)])
        for view in range(12):
            np.testing.assert_array_equal(actual[view, :5], np.zeros((5, 2)))
            np.testing.assert_allclose((actual[view, 5:]-.5)/37, expected, rtol=0, atol=1e-7)

    def packed_fixture(self, group_size=64):
        import mlx.core as mx
        rng = np.random.default_rng(819)
        integers = rng.integers(0, 256, (32, 64), dtype=np.uint32)
        packed = (integers.reshape(32, 16, 4) << np.arange(0, 32, 8, dtype=np.uint32)).sum(
            axis=2, dtype=np.uint32)
        scales = np.full((32, 64 // group_size), .01, dtype=np.float16)
        offsets = np.full((32, 64 // group_size), -1.28, dtype=np.float16)
        bias = rng.normal(0, .1, 32).astype(np.float16)
        weights = {}
        for name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            for suffix, value in (("qweight", packed), ("scales", scales), ("offsets", offsets), ("bias", bias)):
                weights[name + "." + suffix] = mx.array(value)
        unpacked = (integers.astype(np.float32) * np.repeat(scales.astype(np.float32), group_size, axis=1)
                    + np.repeat(offsets.astype(np.float32), group_size, axis=1))
        return weights, unpacked, bias

    def test_fp32_packed_gemm_preserves_contribution_before_bias_cancellation(self):
        import mlx.core as mx
        from rebuild3d_worker.mlx_block import Block, FP32_LINEAR_POLICY
        mx.set_default_device(mx.gpu)
        # Integer values are all zero, so every reconstructed weight is its
        # group's offset. The exact dot product is 1 + 2**-11. Rounding it to
        # FP16 before adding -1 loses the entire remaining contribution.
        weights = {}
        for name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            weights[name + '.qweight'] = mx.zeros((32, 16), dtype=mx.uint32)
            weights[name + '.scales'] = mx.ones((32, 2), dtype=mx.float16)
            weights[name + '.offsets'] = mx.broadcast_to(mx.array([1/32, 2**-16], dtype=mx.float16), (32, 2))
            weights[name + '.bias'] = -mx.ones((32,), dtype=mx.float16)
        block = Block(weights, {"activation": "GELU", "activationApproximation": "none"},
                      group_size=32, linear_policy=FP32_LINEAR_POLICY)
        result = block.linear('attn.qkv', mx.ones((7, 64), dtype=mx.float16))
        mx.eval(result)
        self.assertEqual(result.dtype, mx.float16)
        np.testing.assert_array_equal(np.asarray(result), np.full((7, 32), 2**-11, dtype=np.float16))
        self.assertTrue(all(q.dtype == mx.uint32 for q, _, _ in block.packed.values()))
        self.assertEqual(block.storage()['dispatches']['attn.qkv']['originalWeightParameters'], 2048)

    def test_group32_fp32_policy_matches_independent_numpy_unpack(self):
        import mlx.core as mx
        from rebuild3d_worker.mlx_block import Block, FP32_LINEAR_POLICY
        mx.set_default_device(mx.gpu)
        weights, unpacked, bias = self.packed_fixture(group_size=32)
        block = Block(weights, {"activation": "GELU", "activationApproximation": "none"},
                      group_size=32, linear_policy=FP32_LINEAR_POLICY)
        x = np.random.default_rng(52).normal(0, .2, (7, 64)).astype(np.float16)
        result = block.linear('attn.qkv', mx.array(x))
        mx.eval(result)
        expected = (x.astype(np.float64) @ unpacked.astype(np.float64).T + bias.astype(np.float64)).astype(np.float16)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=.001, atol=1e-6)

    def test_group32_requires_explicit_compute_policy(self):
        from rebuild3d_worker.mlx_block import Block, FP32_LINEAR_POLICY
        weights, _, _ = self.packed_fixture(group_size=32)
        for group, policy in [(32, 'fp16-packed-gemm-fp16-bias-v1'), (32, 'unknown'), (64, FP32_LINEAR_POLICY)]:
            with self.subTest(group=group, policy=policy), self.assertRaisesRegex(ValueError, 'linear compute policy'):
                Block(weights, {"activation": "GELU", "activationApproximation": "none"},
                      group_size=group, linear_policy=policy)

    def test_gpu_linear_matches_independent_numpy_unpack(self):
        import mlx.core as mx
        from rebuild3d_worker.mlx_block import Block
        mx.set_default_device(mx.gpu)
        weights, unpacked, bias = self.packed_fixture()
        block = Block(weights, {"activation": "GELU", "activationApproximation": "none"})
        x = np.random.default_rng(52).normal(0, .2, (7, 64)).astype(np.float16)
        result = block.linear("attn.qkv", mx.array(x))
        mx.eval(result)
        expected = x.astype(np.float32) @ unpacked.T + bias.astype(np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=.015, atol=.008)
        audit = block.storage()["dispatches"]["attn.qkv"]
        self.assertEqual(audit["calls"], 1)
        self.assertEqual(audit["originalWeightParameters"], 2048)
        self.assertEqual(audit["inputShapes"], [[7, 64]])

    def test_missing_packed_weight_never_triggers_online_quantization(self):
        from rebuild3d_worker.mlx_block import Block
        weights, _, _ = self.packed_fixture()
        del weights["attn.qkv.qweight"]
        with self.assertRaisesRegex(ValueError, "online quantization is forbidden"):
            Block(weights, {"activation": "GELU", "activationApproximation": "none"})

    def test_wrong_packing_dtype_is_rejected(self):
        import mlx.core as mx
        from rebuild3d_worker.mlx_block import Block
        weights, _, _ = self.packed_fixture()
        weights["attn.qkv.qweight"] = weights["attn.qkv.qweight"].astype(mx.uint8)
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            Block(weights, {"activation": "GELU", "activationApproximation": "none"})

    def test_unsupported_device_size_and_count_are_rejected(self):
        import mlx.core as mx
        from rebuild3d_worker.mlx_aggregator import Aggregator
        from rebuild3d_worker.mlx_backend import MLXBackend
        with self.assertRaisesRegex(ValueError, "no CPU fallback"):
            MLXBackend(None, None, {}, 518, "cpu")
        with self.assertRaisesRegex(ValueError, "518 and 392"):
            Aggregator(None, None, weight_info={"format": "rebuild3d-vggt-mlx"}, size=336)
        aggregator = Aggregator.__new__(Aggregator)
        aggregator.size = 518
        for count in (2, 13):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "3–12"):
                aggregator(mx.zeros((1, count, 3, 518, 518), dtype=mx.float16))


if __name__ == "__main__":
    unittest.main()
