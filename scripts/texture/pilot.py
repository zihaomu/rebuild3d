"""Small fixed-geometry head pilot, explicitly one-view texture rather than a full seven-view result."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import trimesh
from common import camera_project,full_intrinsic,write_json

p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('baseline',type=Path)
p.add_argument('stage1',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
run=a.stage1/'reproduction-01';g=np.load(run/'fusion/face-provenance.npz')
regions=np.load(a.baseline/'review-region-faces.npz')
ids=np.unique(np.r_[regions['crown'],regions['face']]);faces=g['faces'][ids];v=g['verticesBeforeDisplay']
fullmesh=trimesh.Trimesh(vertices=v,faces=g['faces'],process=False)
records=json.loads((a.inputs/'dataset.json').read_text())['records'];record=records[0]
c=json.loads((run/'inference/cameras.json').read_text())['cameras'][0]
t=json.loads((run/'inference/pixel-transforms.json').read_text())[0]
k=full_intrinsic(record,np.array(c['modelIntrinsics']),t)
projected=camera_project(v,np.array(c['worldToCamera']),k)
xy=projected[:,:2];used=np.unique(faces)
lo=np.maximum(np.floor(xy[used].min(0)-4),0).astype(int)
hi=np.minimum(np.ceil(xy[used].max(0)+4),[4284,5712]).astype(int)
box=tuple(map(int,(*lo,*hi)));im=Image.open(a.inputs/record['image']).convert('RGB').crop(box)
mask=np.array(Image.open(a.inputs/record['mask']).convert('L').crop(box))>200
pixels=np.array(im);pixels[~mask]=[145,120,90]
Image.fromarray(pixels).save(a.output/'basecolor.png')
uv=(xy-lo+.5)/(hi-lo);uv[:,1]=1-uv[:,1]
np.savez_compressed(a.output/'uv.npz',vertexMap=np.arange(len(v)),faces=faces,uv=uv.astype('float32'),originalFaceID=ids,width=im.width,height=im.height)
np.savez_compressed(a.output/'geometry.npz',vertices=v,faces=faces,normals=fullmesh.vertex_normals,completion=g['completion'][ids])
shutil.copy2(run/'fusion/report.json',a.output/'fusion.json')
shutil.copy2(a.baseline/'cameras.json',a.output/'cameras.json')
write_json(a.output/'pilot.json',{'status':'local-front-head-pilot-only','photo':record['name'],'cropFullPixels':box,
    'triangles':len(faces),'originalFaceIDs':'uv.npz/originalFaceID','geometryUnchanged':True,
    'purpose':'Verify real within-face detail and USDZ UV material before full atlas baking',
    'limitations':['One source view and selected head region only; not a complete enhanced reconstruction.','Mask-exterior pixels use a neutral approximate color.']})
