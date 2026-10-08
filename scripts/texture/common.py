import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import psutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class Budget:
    def __init__(self, output, seconds=1200, gib=11):
        self.start = time.monotonic()
        self.stop = threading.Event()
        self.peak = 0
        def monitor():
            with (output/'resources.jsonl').open('w') as f:
                while not self.stop.is_set():
                    rss = psutil.Process().memory_info().rss
                    self.peak = max(self.peak, rss)
                    elapsed = time.monotonic()-self.start
                    f.write(json.dumps({'seconds':elapsed,'rssBytes':rss})+'\n'); f.flush()
                    if elapsed>seconds or rss>gib*1024**3:
                        write_json(output/'failure.json', {'status':'resource-limit','seconds':elapsed,'rssBytes':rss})
                        os._exit(2)
                    self.stop.wait(1)
        threading.Thread(target=monitor, daemon=True).start()
    def finish(self):
        self.stop.set()
        return {'elapsedSeconds':time.monotonic()-self.start,'sampledPeakRSSBytes':self.peak}


def camera_project(points, extrinsic, intrinsic):
    p = points @ extrinsic[:,:3].T + extrinsic[:,3]
    q = p @ intrinsic.T
    return np.c_[q[:,:2]/q[:,2,None],p[:,2]].astype('float32')


def raster(vertices, faces, width, height):
    source = Path(__file__).with_name('raster.cpp')
    cache = Path('build/seven-photo-statue-stage2/tools')
    cache.mkdir(parents=True,exist_ok=True)
    library = cache/('raster-'+sha(source)[:12]+'.dylib')
    if not library.exists():
        subprocess.run(['clang++','-O3','-std=c++17','-dynamiclib',str(source),'-o',str(library)],check=True)
    lib = ctypes.CDLL(str(library.resolve()))
    fn = lib.rasterize
    fn.argtypes = [ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                   ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    vertices=np.ascontiguousarray(vertices,dtype='float32'); faces=np.ascontiguousarray(faces,dtype='int32')
    z=np.empty((height,width),dtype='float32'); ids=np.empty((height,width),dtype='int32')
    b=np.zeros((height,width,2),dtype='float32')
    fn(len(faces),faces.ctypes.data,vertices.ctypes.data,width,height,z.ctypes.data,ids.ctypes.data,b.ctypes.data)
    return z,ids,b


def full_intrinsic(record, model_k, transform):
    # model <- working <- upright full; all resize maps include pixel-center offsets.
    sx,sy=record['workingWidth']/record['uprightWidth'],record['workingHeight']/record['uprightHeight']
    full_to_work=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]])
    return np.linalg.inv(np.asarray(transform['workingToModelPixels'])@full_to_work)@model_k


def resize_intrinsic(k, scale):
    return np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1]])@k
