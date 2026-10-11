import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/v4/compare-inference.py'
SPEC=importlib.util.spec_from_file_location('compare_v4',SCRIPT)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)
THRESHOLDS={'rotationP95Degrees':1,'focalRelativeP95':.02,'depthRelativeMedian':.02,'depthRelativeP95':.05,'validDepthFractionDrop':.005}


class ComparisonTests(unittest.TestCase):
    def fixtures(self,root):
        a,b=root/'a',root/'b';a.mkdir();b.mkdir()
        config={'photoIDs':['1','2','3'],'sourceSHA256':['a','b','c'],'datasetSHA256':'same'}
        for p in (a,b):
            (p/'config.json').write_text(json.dumps(config));(p/'pixel-transforms.json').write_text('[]')
        extrinsics=np.tile(np.c_[np.eye(3),np.zeros(3)],(3,1,1));extrinsics[:,0,3]=[0,-.3,-.6]
        arrays={'extrinsics':extrinsics,'intrinsics':np.tile([[20,0,4],[0,20,4],[0,0,1]],(3,1,1)),
                'depth':np.ones((3,9,9))*2,'confidence':np.ones((3,9,9))*4,
                'rgb':np.ones((3,9,9,3),dtype='uint8'),'masks':np.ones((3,9,9),dtype=bool)}
        np.savez(a/'predictions.npz',**arrays)
        return a,b,arrays

    def test_one_global_similarity_removes_coordinate_choice_only(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,arrays=self.fixtures(Path(d))
            theta=.3;q=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
            t=np.array([.5,-.3,.1]);r=arrays['extrinsics'][:,:,:3].copy()
            arrays['extrinsics'][:,:,:3]=r@q
            arrays['extrinsics'][:,:,3]=(np.einsum('nij,j->ni',r,t)+arrays['extrinsics'][:,:,3])/2
            arrays['depth']/=2;np.savez(b/'predictions.npz',**arrays)
            result=MODULE.compare(a,b,THRESHOLDS)
            self.assertTrue(result['numericGatesPassed'])
            self.assertLess(result['metrics']['projectionDisplacementModelPixels']['max'],1e-8)
            np.testing.assert_allclose(result['sceneSimilarity']['translation'],t,atol=1e-10)
            json.dumps(result)

    def test_independent_per_view_scale_errors_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,arrays=self.fixtures(Path(d))
            arrays['depth'][0]*=1.3;np.savez(b/'predictions.npz',**arrays)
            result=MODULE.compare(a,b,THRESHOLDS)
            self.assertFalse(result['numericGatesPassed']);self.assertFalse(result['gates']['depthP95'])

    def test_photo_order_and_invalid_rotations_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,arrays=self.fixtures(Path(d))
            arrays['extrinsics'][0,0,0]=1.1;np.savez(b/'predictions.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'rotation'):
                MODULE.compare(a,b,THRESHOLDS)
            config=json.loads((b/'config.json').read_text());config['photoIDs'].reverse()
            (b/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'identity/order'):
                MODULE.compare(a,b,THRESHOLDS)

    def relocated_fixtures(self, root):
        a,b,arrays=self.fixtures(root)
        for inference in (a,b):
            # Give each inference its own sibling input directory.
            destination=root/(inference.name+'-job')/'inference'
            destination.parent.mkdir();inference.rename(destination)
            inputs=destination.parent/'inputs';inputs.mkdir()
            records=[]
            for index, source_sha in enumerate(('a','b','c'), 1):
                image,mask=f'{index}.png',f'{index}-mask.png'
                (inputs/image).write_bytes(b'identical-working-image-'+bytes([index]))
                (inputs/mask).write_bytes(b'identical-working-mask-'+bytes([index]))
                records.append({'id':str(index),'sourceSHA256':source_sha,
                                'sourcePath':f'/{inference.name}/{index}.jpg',
                                'image':image,'mask':mask,'width':9,'height':9})
            raw=json.dumps({'records':records}).encode()
            (inputs/'dataset.json').write_bytes(raw)
            config=json.loads((destination/'config.json').read_text())
            config['datasetSHA256']=hashlib.sha256(raw).hexdigest()
            (destination/'config.json').write_text(json.dumps(config))
        a,b=root/'a-job/inference',root/'b-job/inference'
        np.savez(b/'predictions.npz',**arrays)
        return a,b

    def test_relocation_requires_explicit_byte_verified_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=self.relocated_fixtures(Path(d))
            with self.assertRaisesRegex(ValueError,'datasetSHA256'):
                MODULE.compare(a,b,THRESHOLDS)
            result=MODULE.compare(a,b,THRESHOLDS,relocated_inputs=True)
            self.assertTrue(result['numericGatesPassed'])
            self.assertEqual(len(result['inputIdentityProof']['identicalWorkingArtifacts']),6)

    def test_relocation_rejects_changed_working_pixels_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=self.relocated_fixtures(Path(d))
            (b.parent/'inputs/1.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'beyond sourcePath'):
                MODULE.compare(a,b,THRESHOLDS,relocated_inputs=True)
        with tempfile.TemporaryDirectory() as d:
            a,b=self.relocated_fixtures(Path(d))
            path=b.parent/'inputs/dataset.json'
            data=json.loads(path.read_text());data['records'][0]['width']=10
            raw=json.dumps(data).encode();path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'recorded SHA256'):
                MODULE.compare(a,b,THRESHOLDS,relocated_inputs=True)
            config=json.loads((b/'config.json').read_text());config['datasetSHA256']=hashlib.sha256(raw).hexdigest()
            (b/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'beyond sourcePath'):
                MODULE.compare(a,b,THRESHOLDS,relocated_inputs=True)
if __name__=='__main__':unittest.main()
