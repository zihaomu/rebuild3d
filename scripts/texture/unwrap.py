"""Generate reusable UVs without moving any surface point or losing original face IDs."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import trimesh
import xatlas
from common import Budget, sha, write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('geometry',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--size',type=int,default=2048);p.add_argument('--reuse-raw',type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); budget=Budget(a.output)
    d=np.load(a.geometry); v=d['verticesBeforeDisplay'].astype('float32'); f=d['faces'].astype('uint32')
    mesh=trimesh.Trimesh(vertices=v,faces=f,process=False)
    if a.reuse_raw:
        raw=np.load(a.reuse_raw);mapping=raw['vertexMap'];indices=raw['faces'];uv=raw['uv']
        width,height=int(raw['width']),int(raw['height']);charts=int(raw['charts']);utilization=float(raw['utilization'])
    else:
        atlas=xatlas.Atlas(); atlas.add_mesh(v,f,np.asarray(mesh.vertex_normals,dtype='float32'))
        packing=xatlas.PackOptions(); packing.resolution=a.size; packing.padding=4; packing.bilinear=True
        print('Unwrap',len(f),'faces',flush=True)
        atlas.generate(pack_options=packing)
        mapping,indices,uv=atlas[0]
        width,height=atlas.width,atlas.height;charts=atlas.get_mesh_chart_count(0);utilization=atlas.utilization
        np.savez_compressed(a.output/'raw-uv.npz',vertexMap=mapping,faces=indices,uv=uv,
                            width=width,height=height,charts=charts,utilization=utilization)
        assert atlas.atlas_count==1, 'Use a larger atlas or explicit multiple-atlas support'
    assert np.array_equal(mapping[indices],f), 'Face/corner order changed; mapping must be resolved before continuing'
    ab=uv[indices[:,1]]-uv[indices[:,0]]; ac=uv[indices[:,2]]-uv[indices[:,0]]
    area=np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])
    degenerate=np.flatnonzero(area<=1e-15)
    # xatlas can collapse very narrow geometry in UV space. Give each such face
    # its own padded UV island without changing any 3D corner or source identity.
    if len(degenerate):
        extra=int(np.ceil(len(degenerate)/(width//10)))*10
        uv[:,1]=(uv[:,1]*height+extra)/(height+extra)
        height+=extra
        n=np.arange(len(degenerate));x=(n%(width//10))*10+3;y=(n//(width//10))*10+3
        patch=np.stack([np.c_[x,y],np.c_[x+3,y],np.c_[x,y+3]],axis=1).reshape(-1,2)/[width,height]
        start=len(mapping);mapping=np.r_[mapping,f[degenerate].ravel()]
        indices[degenerate]=np.arange(start,start+len(degenerate)*3).reshape(-1,3)
        uv=np.vstack([uv,patch]).astype('float32')
    ab=uv[indices[:,1]]-uv[indices[:,0]];ac=uv[indices[:,2]]-uv[indices[:,0]]
    area=np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]);assert (area>1e-15).all()
    assert np.array_equal(mapping[indices],f)
    np.savez_compressed(a.output/'uv.npz',vertexMap=mapping,faces=indices,uv=uv,
                        originalFaceID=np.arange(len(f)),width=width,height=height)
    write_json(a.output/'report.json',{'xatlasVersion':importlib.metadata.version('xatlas'),'inputSHA256':sha(a.geometry),
        'requestedResolution':a.size,'width':width,'height':height,'xatlasUtilizationBeforeRepair':utilization,
        'chartCount':charts,'paddingPixels':4,'originalVertices':len(v),'uvVertices':len(mapping),
        'repairedDegenerateUVFaces':len(degenerate),
        'triangles':len(f),'allUVFacesHaveArea':True,'surfaceExactlyPreserved':True,'sameFaceCornerOrder':True,
        'minimumUVTriangleArea':float(area.min()/2),**budget.finish()})
    print((a.output/'report.json').read_text(),flush=True)


if __name__=='__main__': main()
