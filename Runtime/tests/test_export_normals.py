"""UV seams and provenance regions must not change exported smooth normals."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom
import trimesh


class ExportNormalsTests(unittest.TestCase):
    def test_compaction_and_uv_splitting_preserve_original_corner_normals(self):
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype='float64')
        faces = np.array([[0,1,2],[0,3,1]], dtype='int32')
        normals = np.array([[0,.6,.8],[0,.6,.8],[0,0,1],[0,1,0]], dtype='float64')
        # Each source region uses only three vertices; the shared smooth normals
        # differ from either triangle's flat normal, exposing cache loss.
        with tempfile.TemporaryDirectory(prefix='rebuild3d-normal-test-') as temporary:
            for split in (False, True):
                with self.subTest(split=split):
                    root = Path(temporary)/str(split)
                    root.mkdir()
                    mapping = faces.ravel() if split else np.arange(4)
                    uvfaces = np.arange(6).reshape(2,3) if split else faces
                    np.savez(root/'geometry.npz', vertices=vertices, faces=faces, normals=normals,
                             completion=np.array([False, True]))
                    np.savez(root/'uv.npz', vertexMap=mapping, faces=uvfaces,
                             uv=np.full((len(mapping),2), .5, dtype='float32'))
                    (root/'fusion.json').write_text(json.dumps({'worldToDisplay':np.eye(4).tolist()}))
                    Image.new('RGB',(4,4),(120,130,140)).save(root/'basecolor.png')
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
                    env['PYTHONDONTWRITEBYTECODE'] = '1'
                    result = subprocess.run([sys.executable, '-B', '-m', 'rebuild3d_worker.export', str(root)],
                                            env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    stage = Usd.Stage.Open(str(root/'model.usdz'))
                    for index, face in enumerate(faces):
                        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f'/Object/region_{index}'))
                        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
                        np.testing.assert_array_equal(np.array(mesh.GetPointsAttr().Get())[indices][0], vertices[face])
                        np.testing.assert_allclose(np.array(mesh.GetNormalsAttr().Get())[indices][0], normals[face], atol=1e-7, rtol=0)
                    scene = trimesh.load(root/'model.glb', force='scene', process=False)
                    for name, face in zip(('learned-depth','silhouette-completion'), faces):
                        mesh = scene.geometry[name]
                        np.testing.assert_allclose(mesh.vertex_normals[mesh.faces][0], normals[face], atol=1e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
