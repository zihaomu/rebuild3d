#!/usr/bin/env python3
"""Diagnostic CPU rendering of the actual exported USDZ, without a display server.

Uses the existing perspective-correct rasterizer. This does not replace the
required SceneKit/application compatibility and display acceptance.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Runtime'))
from rebuild3d_worker.common import camera_project, raster, sha
from rebuild3d_worker.bake import sample


def load_asset(path):
    stage=Usd.Stage.Open(str(path.resolve()))
    if not stage:raise ValueError('USDZ failed to load')
    texture=UsdShade.Shader(stage.GetPrimAtPath('/Object/PhotoMaterial/Texture')).GetInput('file').Get()
    if not texture or Path(texture.path).is_absolute() or '..' in Path(texture.path).parts:
        raise ValueError('Unexpected embedded texture path')
    with zipfile.ZipFile(path) as archive:
        image=np.asarray(Image.open(io.BytesIO(archive.read(texture.path))).convert('RGB'))
    vertices=[];uvs=[];faces=[];normals=[];offset=0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim)
        if np.any(np.asarray(mesh.GetFaceVertexCountsAttr().Get())!=3):raise ValueError('Only triangles supported')
        # Exported meshes contain display-space coordinates and identity transforms.
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
        if not np.allclose(matrix,np.eye(4)):raise ValueError('Unexpected transform; explicit handling required')
        v=np.asarray(mesh.GetPointsAttr().Get(),dtype='float32')
        n=np.asarray(mesh.GetNormalsAttr().Get(),dtype='float32')
        uv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
        if uv.GetInterpolation()!=UsdGeom.Tokens.vertex:raise ValueError('Unexpected UV interpolation')
        vertices.append(v);normals.append(n);uvs.append(np.asarray(uv.Get(),dtype='float32'))
        faces.append(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype='int32').reshape(-1,3)+offset)
        offset+=len(v)
    if not faces:raise ValueError('No mesh in USDZ')
    return np.concatenate(vertices),np.concatenate(faces),np.concatenate(uvs),np.concatenate(normals),image


def render(vertices,faces,uv,normals,texture,extrinsic,intrinsic,width,height,out,prefix):
    _,ids,bary=raster(camera_project(vertices,extrinsic,intrinsic),faces,width,height)
    active=ids>=0;pixels=np.flatnonzero(active)
    color=np.zeros((height*width,3),dtype='uint8');clay=np.zeros_like(color)
    tw,th=texture.shape[1],texture.shape[0]
    flat=ids.ravel();weights=bary.reshape(-1,2)
    light=np.asarray([-.25,-.35,-1.]);light/=np.linalg.norm(light)
    for start in range(0,len(pixels),150000):
        indices=pixels[start:start+150000];tri=faces[flat[indices]]
        w=np.c_[weights[indices],1-weights[indices].sum(1)]
        st=(uv[tri]*w[:,:,None]).sum(1)
        xy=np.c_[st[:,0]*tw-.5,(1-st[:,1])*th-.5]
        color[indices]=np.clip(sample(texture,xy),0,255).round().astype('uint8')
        n=(normals[tri]*w[:,:,None]).sum(1)@extrinsic[:,:3].T
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        shade=(.25+.65*np.maximum(n@light,0))*220
        clay[indices]=shade[:,None].astype('uint8')
    for name,value in [('color',color.reshape(height,width,3)),('clay',clay.reshape(height,width,3)),('mask',active.astype('uint8')*255)]:
        Image.fromarray(value).save(out/f'{prefix}-{name}.png')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    root=a.run;out=root/'renders-cpu';out.mkdir(exist_ok=False)
    model=root/'bundle/model.usdz';v,f,uv,n,texture=load_asset(model)
    display=np.asarray(json.loads((root/'fusion/report.json').read_text())['worldToDisplay'])
    world=(v-display[:3,3])@np.linalg.inv(display[:3,:3]).T
    world_normals=n@display[:3,:3];world_normals/=np.linalg.norm(world_normals,axis=1,keepdims=True)
    cameras=json.loads((root/'bundle/cameras.json').read_text())['cameras']
    for i,c in enumerate(cameras):
        width,height=map(int,c['workingSize'])
        render(world,f,uv,world_normals,texture,np.asarray(c['worldToCamera']),np.asarray(c['workingIntrinsics']),width,height,out,f'view-{i}')
        print('CPU USDZ photo view',i,flush=True)
    lower,upper=v.min(0),v.max(0);center=(lower+upper)/2;extent=(upper-lower).max()
    side=640;focal=side/(2*np.tan(np.radians(42)/2));k=np.array([[focal,0,(side-1)/2],[0,focal,(side-1)/2],[0,0,1.]])
    for i in range(12):
        angle=i*2*np.pi/12;position=center+extent*np.array([np.sin(angle)*1.65,.15,np.cos(angle)*1.65])
        forward=center-position;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        down=np.cross(forward,right);rotation=np.stack((right,down,forward))
        ext=np.c_[rotation,-rotation@position]
        render(v,f,uv,n,texture,ext,k,side,side,out,f'turntable-{i:02d}')
        for mode in ('clay','color','mask'):(out/f'turntable-{i:02d}-{mode}.png').rename(out/f'turntable-{mode}-{i:02d}.png')
    manifest={'renderer':'CPU diagnostic of actual USDZ; not SceneKit/application acceptance',
              'modelSHA256':sha(model),'rendererSHA256':sha(__file__),'triangles':len(f),
              'photoViews':len(cameras),'turntableViews':12,'lighting':'diagnostic Lambert; different from SceneKit',
              'multisampling':False,'nativeAppAcceptance':'pending'}
    (out/'render-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
