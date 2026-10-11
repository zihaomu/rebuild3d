import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

source = Path(__file__).resolve().parents[2]/'scripts/v4/audit-uv-distortion.py'
spec = importlib.util.spec_from_file_location('uv_distortion', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DistortionTests(unittest.TestCase):
    def setUp(self):
        self.surface = np.array([[[0,0,0],[1,0,0],[0,1,0]]], dtype=float)
        self.uv = self.surface[:, :, :2]

    def test_isometry_and_uniform_scale(self):
        rotated = self.uv @ np.array([[0,-1],[1,0]]) + [8,4]
        for pixels in (self.uv, rotated, self.uv*[-1,1]):
            result = module.triangle_metrics(self.surface, pixels)
            np.testing.assert_allclose(result['anisotropy'], 1)
            np.testing.assert_allclose(result['areaDensity'], 1)
        result = module.triangle_metrics(self.surface, rotated*3)
        np.testing.assert_allclose(result['anisotropy'], 1)
        np.testing.assert_allclose(result['areaDensity'], 9)

    def test_anisotropic_scale_and_shear(self):
        result = module.triangle_metrics(self.surface, self.uv*[4,1])
        np.testing.assert_allclose(result['anisotropy'], 4)
        np.testing.assert_allclose(result['areaDensity'], 4)
        result = module.triangle_metrics(self.surface, self.uv @ np.array([[1,0],[1,1]]))
        np.testing.assert_allclose(result['anisotropy'], (3+np.sqrt(5))/2)
        np.testing.assert_allclose(result['areaDensity'], 1)

    def test_surface_orientation_does_not_change_metrics(self):
        tilted = self.surface[:, :, [2,0,1]] + [9,3,7]
        result = module.triangle_metrics(tilted, self.uv)
        np.testing.assert_allclose(result['anisotropy'], 1)
        np.testing.assert_allclose(result['surfaceArea'], .5)

    def test_invalid_triangles_and_weighted_quantiles(self):
        for surface,pixels in ((np.zeros_like(self.surface),self.uv),
                               (self.surface,np.zeros_like(self.uv)),
                               (self.surface,np.array([[[0,0],[1,1],[2,2]]])),
                               (self.surface,self.uv*np.nan)):
            with self.assertRaises(ValueError): module.triangle_metrics(surface,pixels)
        result=module.quantiles(np.array([10,1,2]),np.array([1,6,3]))
        self.assertEqual(result,{'p0':1.,'p50':1.,'p95':10.,'p99':10.,'p100':10.})

    def test_face_ids_restore_order_and_reject_lost_provenance(self):
        faces=np.array([[0,1,2],[1,3,2]])
        uv=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=float)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'uv.npz'
            def write(ids,mapping):
                np.savez(path,vertexMap=mapping,faces=faces[::-1],originalFaceID=ids,
                         uv=uv,width=128,height=256)
            write([1,0],np.arange(4))
            pixels,size=module.load_uv(path,faces)
            np.testing.assert_array_equal(pixels,uv[faces]*[128,256])
            self.assertEqual(size,[128,256])
            for ids,mapping in (([1,1],np.arange(4)),([1,0],np.array([0,2,1,3]))):
                write(ids,mapping)
                with self.assertRaises(ValueError): module.load_uv(path,faces)


if __name__ == '__main__':
    unittest.main()
