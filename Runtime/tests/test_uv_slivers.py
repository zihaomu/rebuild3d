import itertools
import unittest

import numpy as np

from rebuild3d_worker import uv_sampling as module


class SliverTests(unittest.TestCase):
    def test_flattening_preserves_every_edge_for_all_corner_orders(self):
        triangle = np.array([[2, -4, 1], [2, 2, 1], [2, -2, 4]], dtype=float)
        for order in itertools.permutations(range(3)):
            surface = triangle[list(order)]
            flattened, height = module.isometric_triangle(surface)
            self.assertGreater(height, 0)
            expected = np.linalg.norm(surface[:, None]-surface[None, :], axis=2)
            actual = np.linalg.norm(flattened[:, None]-flattened[None, :], axis=2)
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def fixture(self, pixels, surface=None):
        if surface is None:
            surface = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
        triangles = np.array([surface, [[0, 0, 0], [2, 0, 0], [0, 1, 0]]], dtype=float)
        uv = np.array([*pixels, [30, 30], [38, 30], [30, 34]], dtype='float32')/64
        data = dict(uv=uv, faces=np.arange(6, dtype='uint32').reshape(-1, 3),
                    vertexMap=np.arange(6, dtype='uint32'), originalFaceID=np.arange(2),
                    width=np.array(64), height=np.array(64))
        return data, triangles

    def test_collapsed_mapping_gets_isometric_samples_without_changing_corner_ids(self):
        data, surface = self.fixture([[8, 8], [16, 8], [8, 8.1]])
        original = {k: v.copy() for k, v in data.items()}
        result, report = module.repair_sampling(data, surface, 2)
        self.assertEqual(report['repairedFaces'], 1)
        self.assertEqual(report['selectedFaceIndices'], [0])
        self.assertEqual(max(int(result['width']), int(result['height'])), 64)
        pixels = result['uv'][result['faces']]*[result['width'], result['height']]
        metrics = module.triangle_metrics(surface, pixels)
        self.assertLess(metrics['anisotropy'][0], 1.02)  # Rounded rectangular atlas dimensions.
        self.assertGreater(module.altitude(pixels)[0], 1)
        np.testing.assert_array_equal(result['vertexMap'][result['faces']], original['vertexMap'][original['faces']])
        np.testing.assert_array_equal(result['originalFaceID'], original['originalFaceID'])
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])

    def test_genuinely_thin_surface_is_not_mistaken_for_mapping_collapse(self):
        data, surface = self.fixture([[8, 8], [24, 8], [8, 8.08]],
                                    [[0, 0, 0], [4, 0, 0], [0, .02, 0]])
        result, report = module.repair_sampling(data, surface, 2)
        self.assertEqual(report['repairedFaces'], 0)
        self.assertIs(result, data)

    def test_resolved_anisotropic_face_is_not_selected_by_stretch_alone(self):
        data, surface = self.fixture([[2, 2], [42, 2], [2, 4]])
        result, report = module.repair_sampling(data, surface, 2)
        self.assertEqual(report['repairedFaces'], 0)
        self.assertIs(result, data)

    def test_degenerate_surface_is_rejected(self):
        with self.assertRaises(ValueError):
            module.isometric_triangle(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))

    def test_all_small_but_isometric_faces_remain_valid(self):
        data, surface = self.fixture([[8, 8], [8.2, 8], [8, 8.1]])
        data['uv'][3:] = np.array([[30, 30], [30.2, 30], [30, 30.1]])/64
        result, report = module.repair_sampling(data, surface, 2)
        self.assertIs(result, data)
        self.assertEqual(report['repairedFaces'], 0)

    def test_fully_collapsed_atlas_uses_explicit_patch_size(self):
        data, surface = self.fixture([[8, 8], [16, 8], [8, 8.1]])
        data['uv'][3:] = np.array([[30, 30], [38, 30], [30, 30.1]])/64
        result, report = module.repair_sampling(data, surface, 2)
        self.assertEqual(report['repairedFaces'], 2)
        self.assertEqual(report['medianHealthyAreaDensity'], 0)
        self.assertGreater(report['minimumRepairedAltitudeTexels'], 1)


if __name__ == '__main__':
    unittest.main()
