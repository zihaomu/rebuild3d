"""Increase texel density on the same frozen UV charts; never upscale existing colors."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .common import sha,write_json

p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
p.add_argument('--factor',type=float,default=2);p.add_argument('--size',type=int);a=p.parse_args()
assert a.factor>0;a.output.mkdir(parents=True,exist_ok=False)
d=dict(np.load(a.input/'uv.npz'))
factor=a.size/max(int(d['width']),int(d['height'])) if a.size else a.factor
d['width']=np.array(round(int(d['width'])*factor));d['height']=np.array(round(int(d['height'])*factor))
np.savez_compressed(a.output/'uv.npz',**d)
parent=json.loads((a.input/'report.json').read_text())
write_json(a.output/'report.json',{'parentUVSHA256':sha(a.input/'uv.npz'),'parent':parent,
    'width':int(d['width']),'height':int(d['height']),'densityFactor':factor,
    'sameUVGeometry':True,'paddingPixels':max(1,math.floor(parent['paddingPixels']*factor)),
    'requiresFreshPhotographicBake':True})
