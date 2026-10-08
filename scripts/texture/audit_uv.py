"""Quantify UV budgets, actual sampling density and bounded area coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from common import camera_project,write_json,sha

p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('original_geometry',type=Path);a=p.parse_args();root=a.candidate
g=np.load(root/'geometry.npz');u=np.load(root/'uv.npz');original=np.load(a.original_geometry)
assert np.array_equal(g['vertices'],original['verticesBeforeDisplay'])
assert np.array_equal(g['faces'],original['faces']) and np.array_equal(g['completion'],original['completion'])
assert np.array_equal(u['vertexMap'][u['faces']],g['faces'])
v=g['vertices'];f=g['faces'];w,h=int(u['width']),int(u['height']);uv=u['uv']*[w,h]
mesh=trimesh.Trimesh(v,f,process=False);area=mesh.area_faces;labels=np.load(root/'face-selection.npz')['photoIndex']
t=uv[u['faces']];ab=t[:,1]-t[:,0];ac=t[:,2]-t[:,0];texel_area=abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])/2
assert (texel_area>0).all()
edge=v[f[:,1]]-v[f[:,0]];edge2=v[f[:,2]]-v[f[:,0]];length=np.linalg.norm(edge,axis=1)
x=(edge*edge2).sum(1)/length;y=2*area/length
basis=np.zeros((len(f),2,2));basis[:,0,0]=length;basis[:,0,1]=x;basis[:,1,1]=y
jac=np.stack([ab,ac],axis=2)@np.linalg.inv(basis)
sing=np.linalg.svd(jac,compute_uv=False);stretch=sing[:,0]/np.maximum(sing[:,1],1e-12)
source=np.array(Image.open(root/'texture-source.png'));quality=np.array(Image.open(root/'texture-quality.png'))
ids=np.load(root/'atlas-surface.npz')['originalFaceID'];active=ids>=0;count=np.bincount(ids[active],minlength=len(f))
photo=np.bincount(ids[active&(source>0)],minlength=len(f));resolved=count>0;fraction=photo/np.maximum(count,1)
lower=(area*fraction).sum()/area.sum();unresolved=area[~resolved].sum()/area.sum()
stats=lambda x:{'p05':float(np.percentile(x,5)),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'maximum':float(x.max())}
density=[];cams=json.loads((root/'cameras.json').read_text())['cameras']
for i,c in enumerate(cams):
    selected=labels==i
    xy=camera_project(v,np.array(c['worldToCamera']),np.array(c['fullIntrinsics']))[:,:2]
    tri=xy[f[selected]];b=tri[:,1]-tri[:,0];d=tri[:,2]-tri[:,0];projected=abs(b[:,0]*d[:,1]-b[:,1]*d[:,0])/2
    ratio=projected/texel_area[selected]
    density.append({'photo':c['name'],'selectedFaces':int(selected.sum()),'fullPhotoPixelsPerAtlasTexel':stats(ratio)})
write_json(root/'uv-audit.json',{'geometrySHA256':sha(a.original_geometry),'exactWorldSurfaceAndFaceOrder':True,'geometryClassesUnchanged':True,
  'triangles':len(f),'allUVTrianglesHaveArea':True,'atlasSize':[w,h],'occupiedTexels':int(active.sum()),'occupiedTextureFraction':float(active.mean()),
  'uvTriangleAreaTexels':stats(texel_area),'texelsPerRelativeWorldArea':stats(texel_area/area),'uvAnisotropyRatio':stats(stretch),
  'samplingDensity':density,'photoTexelFraction':float(((source>0)&active).sum()/active.sum()),
  'lowConfidencePhotoTexelFraction':float(((quality==1)&active).sum()/active.sum()),'interiorPhotoTexelFraction':float(((quality==2)&active).sum()/active.sum()),
  'appearanceFillTexelFraction':float(((source==0)&active).sum()/active.sum()),
  'surfaceAreaCoverageEstimate':{'photoLower':float(lower),'photoUpperIncludingUnresolvedMicrofaces':float(lower+unresolved),
    'unresolvedSubtexelFaceAreaFraction':float(unresolved),'unresolvedSubtexelFaces':int((~resolved).sum()),
    'method':'Per-face valid atlas texel fraction weighted by original 3D triangle area; zero-raster-texel faces reported as unresolved, never silently promoted to supported.'},
  'limits':'Density is geometric projection sampling potential, not a measured image resolution or calibrated length. Extreme UV stretch includes tiny repaired islands.'})
print((root/'uv-audit.json').read_text())
