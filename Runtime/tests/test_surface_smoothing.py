"""Analytic openings and sheet boundaries must survive mesh smoothing."""
import unittest

import numpy as np
import trimesh

from rebuild3d_worker.surface_smoothing import (
    silhouette_dominated_vertices, smooth_preserving_features,
)


class SurfaceSmoothingTests(unittest.TestCase):
    def test_thin_ring_preserves_its_opening_and_sharp_rims(self):
        mesh = trimesh.creation.annulus(r_min=.98, r_max=1., height=.03, sections=48)
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        unrestricted = mesh.copy()
        trimesh.smoothing.filter_taubin(unrestricted, lamb=.5, nu=.53, iterations=3)
        self.assertGreater(np.max(np.linalg.norm(unrestricted.vertices - vertices, axis=1)), .001)
        policy = smooth_preserving_features(mesh, np.ones(len(vertices), dtype=bool))
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(policy['pinnedVertices'], len(vertices))
        # A ray through the central aperture cannot be blocked by a new cap.
        self.assertGreater(np.linalg.norm(mesh.triangles_center[:, :2], axis=1).min(), .97)

    def test_open_sheet_keeps_boundary_but_smooths_interior_noise(self):
        axis = np.linspace(-1, 1, 11)
        x, y = np.meshgrid(axis, axis)
        z = .015 * np.cos(3 * np.pi * x) * np.cos(3 * np.pi * y)
        vertices = np.c_[x.ravel(), y.ravel(), z.ravel()]
        faces = []
        for row in range(10):
            for col in range(10):
                a = row * 11 + col
                faces.extend([[a, a + 1, a + 12], [a, a + 12, a + 11]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        boundary = (abs(vertices[:, 0]) == 1) | (abs(vertices[:, 1]) == 1)
        policy = smooth_preserving_features(mesh, np.ones(len(vertices), dtype=bool))
        np.testing.assert_array_equal(mesh.vertices[boundary], vertices[boundary])
        np.testing.assert_array_equal(mesh.faces, faces)
        self.assertEqual(policy['sharpEdges'], 0)
        self.assertLess(np.std(mesh.vertices[~boundary, 2]), np.std(vertices[~boundary, 2]))
        self.assertFalse(mesh.is_watertight)

    def test_smooth_closed_surface_is_not_frozen(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        before, faces = mesh.vertices.copy(), mesh.faces.copy()
        policy = smooth_preserving_features(mesh, np.ones(len(before), dtype=bool))
        self.assertEqual(policy['pinnedVertices'], 0)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(np.isfinite(mesh.vertices).all())
        self.assertGreater(np.linalg.norm(mesh.vertices - before), 0)
        np.testing.assert_array_equal(mesh.faces, faces)

    def test_depth_dominated_closed_features_keep_original_smoothing(self):
        mesh = trimesh.creation.box()
        original = mesh.copy()
        trimesh.smoothing.filter_taubin(original, lamb=.5, nu=.53, iterations=3)
        policy = smooth_preserving_features(mesh, np.zeros(len(mesh.vertices), dtype=bool))
        np.testing.assert_array_equal(mesh.vertices, original.vertices)
        self.assertEqual(policy['pinnedVertices'], 0)

    def test_hull_classification_obeys_intersection_not_object_scale(self):
        # Left surface is depth-constrained, right is hull-constrained. A common
        # translation/scale must not change which field supplies the boundary.
        grid = np.indices((5, 5, 5))[0].astype(float) - 2
        hull, field = grid, np.maximum(grid, -grid)
        samples = np.array([[.5, 2., 2.], [3.5, 2., 2.]])
        for scale in (1., .001, 10.):
            origin = np.array([2., -3., 5.])
            protected = silhouette_dominated_vertices(origin + samples * scale,
                origin, scale, field * scale, hull * scale)
            np.testing.assert_array_equal(protected, [False, True])


if __name__ == '__main__':
    unittest.main()
