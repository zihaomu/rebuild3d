"""Keep final UV triangles disjoint while preserving every original surface corner."""
import ctypes
import math
import os
from pathlib import Path
import time

import numpy as np

from .common import sha


def library_path():
    return Path(os.environ.get('REBUILD3D_UV_LAYOUT_LIBRARY', str(Path(__file__).with_suffix('.dylib')))).resolve()


def validate(data):
    uv = np.asarray(data['uv'])
    faces = np.asarray(data['faces'])
    width, height = int(data['width']), int(data['height'])
    if width<1 or height<1 or width>2147483647 or height>2147483647:
        raise ValueError('Invalid UV atlas dimensions')
    if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all() or not ((uv>=0)&(uv<=1)).all():
        raise ValueError('UV coordinates must be finite and inside the atlas')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or len(faces)>2147483647 or faces.dtype.kind not in 'iu':
        raise ValueError('Expected nonempty integer UV triangles')
    if faces.min()<0 or faces.max()>=len(uv):
        raise ValueError('Invalid UV face index')
    return uv,faces,width,height


def overlapping_faces(data):
    uv,faces,width,height = validate(data)
    path = library_path()
    if not path.is_file():
        raise RuntimeError('Managed UV layout component is missing; install or repair the runtime.')
    native = ctypes.CDLL(str(path))
    function = native.r3d_uv_overlap_mask
    function.argtypes = [ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                         ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_double)]
    function.restype = ctypes.c_int
    triangles = np.ascontiguousarray(uv.astype('float64')[faces]*[width,height])
    affected = np.zeros(len(faces),dtype='uint8')
    pairs,maximum = ctypes.c_uint64(),ctypes.c_double()
    result = function(len(faces),triangles.ctypes.data,width,height,affected.ctypes.data,
                      ctypes.byref(pairs),ctypes.byref(maximum))
    if result:
        raise RuntimeError(f'Continuous UV overlap detection failed ({result})')
    return np.flatnonzero(affected), {'overlappingTrianglePairs':pairs.value,
        'maximumIntersectionTexelSquared':maximum.value,'affectedFaces':int(affected.sum())}


def repair_overlaps(data, padding):
    started = time.monotonic()
    uv,faces,width,height = validate(data)
    mapping = np.asarray(data['vertexMap'])
    if padding<1 or mapping.ndim!=1 or len(mapping)!=len(uv) or mapping.dtype.kind not in 'iu' or mapping.min()<0:
        raise ValueError('Invalid UV padding or vertex map')
    affected,before = overlapping_faces(data)
    detected = time.monotonic()
    report = {'policy':'isolate-all-overlapping-faces-v1','nativeLibrarySHA256':sha(library_path()),
              'intersectionToleranceSquareTexels':1e-8,'before':before,
              'repairedOverlappingFaces':len(affected),'repairedOverlappingFaceIDs':affected.tolist(),
              'addedStripHeightBeforeScaling':0,'texelScaleAfterAddingIslands':1.0,
              'paddingPixels':padding,'detectSeconds':detected-started}
    if not len(affected):
        report.update(after=before.copy(),repairAndVerifySeconds=0.0,elapsedSeconds=time.monotonic()-started)
        return data,report
    faces,mapping = faces.copy(),mapping.copy()
    original_corners = mapping[faces].copy()
    pixels = uv.astype('float64')*[width,height]
    patches = []
    x=y=row_height=0
    for face_id in affected:
        triangle = pixels[faces[face_id]]
        local = triangle-triangle.min(axis=0)
        cell_width,cell_height = np.ceil(local.max(axis=0)).astype(int)+2*padding+2
        if cell_width>width:
            raise ValueError('An overlapping triangle cannot fit its isolated UV row without distortion')
        if x+cell_width>width:
            x,y,row_height=0,y+row_height,0
        patches.append(local+[x+padding+1,y+padding+1])
        x+=int(cell_width);row_height=max(row_height,int(cell_height))
    strip_height = y+row_height+padding+1
    pixels[:,1]+=strip_height
    first = len(mapping)
    mapping = np.concatenate([mapping,original_corners[affected].ravel()])
    faces[affected] = np.arange(first,first+3*len(affected)).reshape(-1,3)
    combined = np.vstack([pixels,np.concatenate(patches)])/[width,height+strip_height]
    factor = max(width,height)/max(width,height+strip_height)
    result = dict(data,uv=combined.astype('float32'),faces=faces,vertexMap=mapping,
                  width=np.array(round(width*factor)),height=np.array(round((height+strip_height)*factor)))
    if not np.array_equal(mapping[faces],original_corners):
        raise ValueError('UV repair changed original face/corner identities')
    triangle = result['uv'].astype('float64')[faces]
    ab,ac=triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]
    if not (abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])>0).all():
        raise ValueError('UV repair produced a degenerate triangle')
    remaining,after=overlapping_faces(result)
    if len(remaining):
        raise ValueError('UV repair left overlapping faces; refusing ambiguous texture sources')
    report.update(after=after,addedStripHeightBeforeScaling=strip_height,
        texelScaleAfterAddingIslands=factor,paddingPixels=max(1,math.floor(padding*factor)),
        originalFaceCornerOrderPreserved=True,repairAndVerifySeconds=time.monotonic()-detected,
        elapsedSeconds=time.monotonic()-started)
    return result,report
