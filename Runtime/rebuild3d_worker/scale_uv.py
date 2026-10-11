"""Set the final atlas budget and remove overlapping UVs before photographic baking."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .common import sha,write_json
from .uv_layout import repair_overlaps, overlapping_faces
from .uv_sampling import repair_sampling

p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
p.add_argument('--factor',type=float,default=2);p.add_argument('--size',type=int)
p.add_argument('--geometry',type=Path,required=True,help='Matching fusion face-provenance.npz')
a=p.parse_args()
assert a.factor>0;a.output.mkdir(parents=True,exist_ok=False)
d=dict(np.load(a.input/'uv.npz'))
factor=a.size/max(int(d['width']),int(d['height'])) if a.size else a.factor
d['width']=np.array(round(int(d['width'])*factor));d['height']=np.array(round(int(d['height'])*factor))
parent=json.loads((a.input/'report.json').read_text())
padding=max(1,math.floor(parent['paddingPixels']*factor))
d,repair=repair_overlaps(d,padding)
with np.load(a.geometry) as geometry:
    ids=d['originalFaceID']
    if not np.array_equal(np.sort(ids),np.arange(len(geometry['faces']))):
        raise ValueError('UV original face identities are incomplete')
    if not np.array_equal(d['vertexMap'][d['faces']],geometry['faces'][ids]):
        raise ValueError('UV original corner map does not match fusion geometry')
    surface=geometry['verticesBeforeDisplay'][geometry['faces'][ids]]
d,sampling=repair_sampling(d,surface,repair['paddingPixels'])
sampling.update(policy='isometric-subpixel-slivers-v1',
    criteria={'anisotropyAbove':4.,'altitudeBelowTexels':1.,'patchAltitudeBeforeScaling':2.5},
    repairedOriginalFaceIDs=ids[sampling['selectedFaceIndices']].tolist())
if sampling['repairedFaces']:
    affected,layout=overlapping_faces(d)
    if len(affected):
        raise ValueError('UV sampling repair introduced overlap; refusing ambiguous texture sources')
    sampling['afterOverlap']=layout
np.savez_compressed(a.output/'uv.npz',**d)
write_json(a.output/'report.json',{'parentUVSHA256':sha(a.input/'uv.npz'),'parent':parent,
    'geometrySHA256':sha(a.geometry),'samplingRepair':sampling,
    'width':int(d['width']),'height':int(d['height']),'densityFactor':factor,
    'effectiveDensityFactor':factor*repair['texelScaleAfterAddingIslands']*sampling['texelScale'],
    'sameUVGeometry':repair['repairedOverlappingFaces']==0 and sampling['repairedFaces']==0,'surfaceExactlyPreserved':True,
    'paddingPixels':max(1,math.floor(repair['paddingPixels']*sampling['texelScale'])),'overlapRepair':repair,
    'requiresFreshPhotographicBake':True})
