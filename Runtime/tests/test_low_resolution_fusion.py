"""Low-resolution consensus must preserve supported surfaces and reject false fill."""
import unittest

import numpy as np

from rebuild3d_worker.fuse import resolve_depth_field


class LowResolutionFusionTests(unittest.TestCase):
    def test_two_near_views_preserve_a_thin_surface_against_one_far_prediction(self):
        # Two independent depth estimates place the sample just inside a thin
        # surface. A third, distant depth estimate incorrectly votes free space.
        near = (np.array([-.35], dtype='float32'), np.array([2.], dtype='float32'),
                np.array([2], dtype='uint8'))
        args = (np.array([.65], dtype='float32'), np.array([3.], dtype='float32'),
                np.array([-1.], dtype='float32'), np.array([-.5], dtype='float32'))
        original, _ = resolve_depth_field(*args)
        revised, policy = resolve_depth_field(*args, near=near)
        self.assertGreater(original[0], 0)
        self.assertLess(revised[0], 0)
        self.assertEqual(policy['changedSignVoxelsInsideHull'], 1)

    def test_one_near_view_cannot_create_a_surface(self):
        field, _ = resolve_depth_field(np.array([.8]), np.array([3.]),
            np.array([-1.]), np.array([-.5]),
            near=(np.array([-.2]), np.array([1.]), np.array([1])))
        self.assertGreater(field[0], 0)

    def test_photographed_background_overrides_near_depth_agreement(self):
        field, policy = resolve_depth_field(np.array([.6]), np.array([3.]),
            np.array([-1.]), np.array([.3]),
            near=(np.array([-.4]), np.array([2.]), np.array([2])))
        self.assertEqual(field[0], .3)
        self.assertEqual(policy['changedSignVoxelsInsideHull'], 0)

    def test_near_free_space_is_not_promoted_to_solid(self):
        field, _ = resolve_depth_field(np.array([1.2]), np.array([3.]),
            np.array([-1.]), np.array([-.5]),
            near=(np.array([.2]), np.array([2.]), np.array([2])))
        self.assertGreater(field[0], 0)

    def test_no_consensus_preserves_original_field_and_unknown_frusta(self):
        args = (np.array([-.6, .5, 0, 0], dtype='float32'),
                np.array([2., 1., 0, 0], dtype='float32'),
                np.array([-1., -1., -1., 1.], dtype='float32'),
                np.array([-1., -1., -.25, -1.], dtype='float32'))
        expected = np.array([-.3, .5, -.25, 1.], dtype='float32')
        original, policy = resolve_depth_field(*args)
        np.testing.assert_array_equal(original, expected)
        self.assertEqual(policy['policy'], 'confidence-tsdf-v1')
        no_consensus = (np.zeros(4, dtype='float32'), np.zeros(4, dtype='float32'),
                        np.zeros(4, dtype='uint8'))
        revised, _ = resolve_depth_field(*args, near=no_consensus)
        np.testing.assert_array_equal(revised, original)


if __name__ == '__main__':
    unittest.main()
