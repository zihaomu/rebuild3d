"""Measure color jumps on both sides of original shared mesh edges in baked texture space."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import trimesh
from bake import sample
from common import write_json

p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--reference',type=Path);a=p.parse_args()
d=np.load(a.candidate/'geometry.npz');u=np.load(a.candidate/'uv.npz')
mesh=trimesh.Trimesh(vertices=d['vertices'],faces=d['faces'],process=False)
adj=mesh.face_adjacency;edges=mesh.face_adjacency_edges
labels=np.load(a.candidate/'face-selection.npz')['photoIndex']
im=np.array(Image.open(a.candidate/'basecolor.png').convert('RGB'))
colors=[]
for side in [0,1]:
    f=adj[:,side];corners=d['faces'][f]
    on_edge=(corners==edges[:,0,None])|(corners==edges[:,1,None])
    weights=np.where(on_edge,.49,.02)
    uv=(u['uv'][u['faces'][f]]*weights[:,:,None]).sum(1)
    xy=uv*[im.shape[1],-im.shape[0]]+[-.5,im.shape[0]-.5]
    rgb=sample(im,xy).astype('float32')/255.
    colors.append(cv2.cvtColor(rgb.reshape(-1,1,3),cv2.COLOR_RGB2LAB).reshape(-1,3))
delta=np.linalg.norm(colors[0]-colors[1],axis=1)
both=(labels[adj]>=0).all(1);boundary=both&(labels[adj[:,0]]!=labels[adj[:,1]])
def stats(mask):
    v=delta[mask]
    return {'edges':int(mask.sum()),'meanDeltaE76':float(v.mean()) if len(v) else None,
            'p95DeltaE76':float(np.percentile(v,95)) if len(v) else None,
            'fractionAbove20':float((v>20).mean()) if len(v) else None}
report={'measurement':'CIE Lab deltaE76 sampled just inside adjacent faces at shared edge midpoint; includes real color detail, not a pure seam oracle',
        'allSharedEdges':stats(np.ones(len(adj),dtype=bool)),'photoViewBoundaries':stats(boundary)}
if a.reference:
    ref=np.load(a.reference/'seam-edges.npz');assert np.array_equal(ref['adjacency'],adj)
    report['onReferenceBoundarySet']=stats(ref['boundary'])
np.savez_compressed(a.candidate/'seam-edges.npz',adjacency=adj,edges=edges,boundary=boundary,deltaE76=delta)
write_json(a.candidate/'seam-report.json',report);print(json.dumps(report,indent=2))
