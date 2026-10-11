#!/usr/bin/env python3
"""Measure xatlas chart/packing phases and bounded spatial partition candidates.

The partition changes UV seams only. Vertex positions, original face ordering,
vertex normals and the total atlas budget are preserved. Final texture review
is mandatory; this experiment never replaces the application configuration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"Runtime"))


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--extension-dir',type=Path,required=True)
    p.add_argument('--chunk-face-budget',type=int,default=0)
    p.add_argument('--block-align',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.extension_dir.resolve()))
    import xatlas
    from rebuild3d_worker.uv_partition import partition_faces
    if not Path(xatlas.__file__).resolve().is_relative_to(a.extension_dir.resolve()):raise ValueError('Wrong xatlas extension')
    if not hasattr(xatlas.Atlas,'generate_profile'):raise ValueError('Split-phase measurement extension required')
    started=time.monotonic()
    g=np.load(a.geometry);v=g['verticesBeforeDisplay'].astype('float32');f=g['faces'].astype('uint32')
    normals=np.asarray(trimesh.Trimesh(vertices=v,faces=f,process=False).vertex_normals,dtype='float32')
    groups=partition_faces(v,f,a.chunk_face_budget)
    atlas=xatlas.Atlas();remaps=[]
    for ids in groups:
        used,inverse=np.unique(f[ids],return_inverse=True)
        atlas.add_mesh(v[used],inverse.reshape(-1,3).astype('uint32'),normals[used]);remaps.append(used)
    prep=time.monotonic()-started
    chart=xatlas.ChartOptions();packing=xatlas.PackOptions()
    packing.resolution=2048;packing.padding=4;packing.bilinear=True;packing.blockAlign=a.block_align
    config={'geometrySHA256':digest(a.geometry),'extensionSHA256':digest(xatlas.__file__),
            'scriptSHA256':digest(__file__),'partitionSourceSHA256':digest(Path(__file__).resolve().parents[2]/'Runtime/rebuild3d_worker/uv_partition.py'),'chunkFaceBudget':a.chunk_face_budget,'meshChunks':len(groups),
            'blockAlign':a.block_align,'resolution':2048,'padding':4,'bilinear':True,
            'xatlasChartDefaults':{k:getattr(chart,k) for k in dir(chart) if not k.startswith('_')},
            'scope':'Diagnostic UV experiment; not final full-task timing or visual acceptance'}
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'event':'generate-start','triangles':len(f),'meshChunks':len(groups)}),flush=True)
    timings=atlas.generate_profile(chart,packing)
    if atlas.atlas_count!=1:raise ValueError('Expected one atlas')
    maps=[];uvs=[];indices=np.zeros(f.shape,dtype='uint32');offset=0
    for i,ids in enumerate(groups):
        mapping,local_faces,uv=atlas[i]
        maps.append(remaps[i][mapping]);uvs.append(uv)
        indices[ids]=local_faces+offset;offset+=len(mapping)
    mapping=np.concatenate(maps);uv=np.concatenate(uvs)
    if not np.array_equal(mapping[indices],f):raise ValueError('Original face/corner order changed')
    chart_count=sum(atlas.get_mesh_chart_count(i) for i in range(len(groups)))
    np.savez_compressed(a.output/'raw-uv.npz',vertexMap=mapping,faces=indices,uv=uv,
                        width=atlas.width,height=atlas.height,charts=chart_count,utilization=atlas.utilization)
    report={**config,**timings,'prepareSeconds':prep,'elapsedSeconds':time.monotonic()-started,
            'width':atlas.width,'height':atlas.height,'charts':chart_count,'utilization':atlas.utilization,
            'originalFaceCornerOrderPreserved':True,'triangleCount':len(f),'uvVertexCount':len(mapping)}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
