"""UV parallelism must preserve geometry, face identities and source ordering."""
import unittest

import numpy as np
import trimesh

from rebuild3d_worker.uv_partition import generate_uv, partition_faces


class UVPartitionTests(unittest.TestCase):
    def mesh(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        # Non-spatial face order exposes incorrect concatenation of chunk faces.
        faces = mesh.faces[np.random.default_rng(61).permutation(len(mesh.faces))]
        mesh = trimesh.Trimesh(mesh.vertices, faces, process=False)
        return (np.asarray(mesh.vertices, dtype='float32'), np.asarray(mesh.faces, dtype='uint32'),
                np.asarray(mesh.vertex_normals, dtype='float32'))

    def test_partition_is_complete_bounded_and_deterministic(self):
        vertices, faces, _ = self.mesh()
        groups = partition_faces(vertices, faces, 37)
        self.assertTrue(all(0 < len(group) <= 37 for group in groups))
        np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(faces)))
        repeat = partition_faces(vertices.copy(), faces.copy(), 37)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(groups, repeat)))
        self.assertEqual(len(partition_faces(vertices, faces, 0)), 1)

    def test_real_atlas_keeps_every_original_corner_and_vertex(self):
        vertices, faces, normals = self.mesh()
        originals = [x.copy() for x in (vertices, faces, normals)]
        raw, phases = generate_uv(vertices, faces, normals, 512, 80)
        np.testing.assert_array_equal(raw['vertexMap'][raw['faces']], faces)
        np.testing.assert_array_equal(vertices[raw['vertexMap'][raw['faces']]], vertices[faces])
        self.assertTrue(np.isfinite(raw['uv']).all())
        self.assertGreater(raw['meshChunks'], 1)
        self.assertGreater(phases['generateSeconds'], 0)
        for actual, reference in zip((vertices, faces, normals), originals):
            np.testing.assert_array_equal(actual, reference)

    def test_invalid_surface_or_budget_fails_before_native_work(self):
        vertices, faces, _ = self.mesh()
        with self.assertRaisesRegex(ValueError, 'negative'):
            partition_faces(vertices, faces, -1)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            partition_faces(vertices, faces[:0], 20)
        vertices[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            partition_faces(vertices, faces, 20)


if __name__ == '__main__':
    unittest.main()
