"""Surface orientation must follow the known scalar field, including open patches."""
import unittest

import numpy as np

from rebuild3d_worker.fuse import extract_surface


class FusionOrientationTests(unittest.TestCase):
    def test_closed_sphere_faces_point_toward_known_exterior(self):
        axis = np.linspace(-1, 1, 32)
        xyz = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
        radius = .65
        mesh = extract_surface(np.linalg.norm(xyz, axis=-1) - radius,
                               np.full(3, -1.), axis[1] - axis[0])
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(mesh.volume, 0)
        outward = np.sum(mesh.triangles_center * mesh.face_normals, axis=1)
        self.assertTrue(np.all(outward > 0))

    def test_open_plane_stays_camera_facing_after_normal_repair_and_translation(self):
        # This patch is intentionally open. Its exterior is x > 0.13, so an
        # exterior camera must see positive-facing normals without using volume.
        axis = np.linspace(-1, 1, 20)
        xyz = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
        for offset in (np.zeros(3), np.array([10., -6., 3.])):
            with self.subTest(offset=offset.tolist()):
                mesh = extract_surface(xyz[..., 0] - .13,
                                       np.full(3, -1.) + offset, axis[1] - axis[0])
                self.assertFalse(mesh.is_watertight)
                mesh.fix_normals(multibody=True)
                self.assertTrue(mesh.is_winding_consistent)
                camera = offset + [3., 0., 0.]
                directions = camera - mesh.triangles_center
                facing = np.sum(mesh.face_normals * directions, axis=1)
                self.assertTrue(np.all(facing > 0), 'Visible open patch would lose photographic texture')
                np.testing.assert_allclose(mesh.face_normals, np.tile([1., 0., 0.], (len(mesh.faces), 1)), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
