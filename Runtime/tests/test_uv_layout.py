"""Continuous UV overlap repair must preserve surface identities and atlas budget."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rebuild3d_worker.uv_layout import overlapping_faces, repair_overlaps


class UVLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='rebuild3d-uv-layout-')
        cls.library = Path(cls.temporary.name)/'uv_layout.dylib'
        source = Path(__file__).resolve().parents[1]/'rebuild3d_worker/uv_layout.cpp'
        subprocess.run(['xcrun','clang++','-O2','-std=c++17','-dynamiclib',str(source),'-o',str(cls.library)],
                       check=True,capture_output=True,text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.environment = patch.dict(os.environ, {'REBUILD3D_UV_LAYOUT_LIBRARY':str(self.library)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def data(self, triangles):
        uv = np.array(triangles,dtype='float32').reshape(-1,2)/128
        faces = np.arange(len(uv),dtype='uint32').reshape(-1,3)
        return {'uv':uv,'faces':faces,'vertexMap':np.arange(len(uv))+10,
                'originalFaceID':np.arange(len(faces))*7+11,'width':np.array(128),'height':np.array(128)}

    def test_overlaps_inside_same_component_are_detected(self):
        data = self.data([[[1,1],[5,1],[1,5]],[[1,1],[3,1],[1,3]]])
        data['faces'][1,0] = 0
        affected,report = overlapping_faces(data)
        np.testing.assert_array_equal(affected,[0,1])
        self.assertAlmostEqual(report['maximumIntersectionTexelSquared'],2)

    def test_shared_edge_and_disjoint_triangles_are_not_overlaps(self):
        data = self.data([[[1,1],[5,1],[1,5]],[[5,1],[5,5],[1,5]],[[8,8],[10,8],[8,10]]])
        copies = {key:value.copy() for key,value in data.items()}
        result,report = repair_overlaps(data,4)
        self.assertEqual(report['before']['overlappingTrianglePairs'],0)
        for key in data:
            np.testing.assert_array_equal(data[key],copies[key])
            np.testing.assert_array_equal(result[key],copies[key])

    def test_repair_preserves_all_original_corners_sources_and_maximum_size(self):
        data = self.data([[[1,1],[5,1],[1,5]],[[1,1],[5,1],[1,5]],[[20,20],[25,20],[20,25]]])
        original = {key:value.copy() for key,value in data.items()}
        result,report = repair_overlaps(data,4)
        self.assertEqual(report['repairedOverlappingFaces'],2)
        self.assertEqual(report['after']['overlappingTrianglePairs'],0)
        self.assertEqual(max(int(result['width']),int(result['height'])),128)
        np.testing.assert_array_equal(result['vertexMap'][result['faces']],data['vertexMap'][data['faces']])
        np.testing.assert_array_equal(result['originalFaceID'],data['originalFaceID'])
        for key in data:
            np.testing.assert_array_equal(data[key],original[key])
        again,_ = repair_overlaps(data,4)
        for key in result:
            np.testing.assert_array_equal(result[key],again[key])

    def test_invalid_uv_indices_and_missing_library_fail_explicitly(self):
        data = self.data([[[1,1],[5,1],[1,5]]])
        data['uv'][0,0] = np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            repair_overlaps(data,4)
        data = self.data([[[1,1],[5,1],[1,5]]])
        data['faces'][0,0] = 999
        with self.assertRaisesRegex(ValueError,'index'):
            repair_overlaps(data,4)
        data = self.data([[[1,1],[5,1],[1,5]]])
        with patch.dict(os.environ,{'REBUILD3D_UV_LAYOUT_LIBRARY':str(self.library)+'.missing'}):
            with self.assertRaisesRegex(RuntimeError,'missing'):
                repair_overlaps(data,4)


if __name__ == '__main__':
    unittest.main()
