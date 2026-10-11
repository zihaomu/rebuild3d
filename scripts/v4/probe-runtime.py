#!/usr/bin/env python3
"""Smoke-test packaged CPU dependencies in an isolated packaged interpreter.

No model inference or application UI is executed. MLX metadata/library presence
is checked, but Metal execution remains an independent acceptance requirement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROBE = r'''
import importlib
import importlib.metadata
import importlib.util
import ctypes
import io
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / 'worker'))
modules = {name: importlib.import_module(name) for name in
           ['numpy', 'torch', 'torchvision', 'PIL.Image', 'cv2', 'scipy', 'trimesh', 'xatlas', 'pxr.Usd', 'pxr.UsdGeom', 'pxr.UsdShade', 'rebuild3d_worker.failures', 'rebuild3d_worker.uv_layout', 'rebuild3d_worker.surface_smoothing']}
paths = {name: module.__file__ for name, module in modules.items()}
assert all(Path(path).resolve().is_relative_to(root) for path in [sys.prefix, *paths.values()])
assert all(Path(path).resolve().is_relative_to(root) for path in sys.path)
np, atlas_module = modules['numpy'], modules['xatlas']
vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype='float32')
faces = np.array([[0,1,2], [0,2,3]], dtype='uint32')
atlas = atlas_module.Atlas()
atlas.add_mesh(vertices, faces)
options = atlas_module.PackOptions()
options.resolution = 32
atlas.generate_profile(atlas_module.ChartOptions(), options)
mapping, output_faces, uv = atlas[0]
assert atlas.atlas_count == 1 and np.array_equal(mapping[output_faces], faces)
assert np.isfinite(uv).all()
image = modules['PIL.Image'].new('RGB', (2,2), (10,20,30))
buffer = io.BytesIO()
image.save(buffer, format='PNG')
assert modules['PIL.Image'].open(io.BytesIO(buffer.getvalue())).size == (2,2)
assert modules['cv2'].cvtColor(np.array(image), modules['cv2'].COLOR_RGB2BGR)[0,0].tolist() == [30,20,10]
mesh = modules['trimesh'].Trimesh(vertices=vertices, faces=faces, process=False)
assert np.isfinite(mesh.vertex_normals).all()
ring = modules['trimesh'].creation.annulus(r_min=.98, r_max=1., height=.03, sections=48)
ring_before = ring.vertices.copy()
surface = modules['rebuild3d_worker.surface_smoothing']
surface.smooth_preserving_features(ring, np.ones(len(ring.vertices), dtype=bool))
assert np.array_equal(ring_before, ring.vertices)
torch = modules['torch']
assert torch.equal(torch.ones((2,2)) @ torch.eye(2), torch.ones((2,2)))
assert modules['torchvision'].ops.nms(torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]), torch.tensor([.9,.8]), .5).tolist() == [0]
stage = modules['pxr.Usd'].Stage.CreateInMemory()
usd_mesh = modules['pxr.UsdGeom'].Mesh.Define(stage, '/Mesh')
usd_mesh.CreatePointsAttr(vertices.tolist())
usd_mesh.CreateFaceVertexCountsAttr([3,3])
usd_mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
assert len(usd_mesh.GetFaceVertexIndicesAttr().Get()) == 6
failures = modules['rebuild3d_worker.failures']
assert failures.failure_fields(RuntimeError('[malloc] Unable to allocate 4096 bytes.'), failures.MLX_BACKEND)['status'] == 'resource-limit'
layout = modules['rebuild3d_worker.uv_layout']
layout_input = {'uv':np.array([[.1,.1],[.3,.1],[.1,.3]]*2, dtype='float32'),
                'faces':np.arange(6, dtype='uint32').reshape(2,3), 'vertexMap':np.arange(6),
                'originalFaceID':np.array([17,31]), 'width':np.array(64), 'height':np.array(64)}
repaired, layout_report = layout.repair_overlaps(layout_input, 2)
assert layout_report['before']['affectedFaces'] == 2 and layout_report['after']['overlappingTrianglePairs'] == 0
assert np.array_equal(repaired['vertexMap'][repaired['faces']], layout_input['faces'])
assert np.array_equal(repaired['originalFaceID'], layout_input['originalFaceID'])
mlx_core = Path(importlib.util.find_spec('mlx.core').origin)
assert mlx_core.resolve().is_relative_to(root)
assert (mlx_core.parent / 'lib/mlx.metallib').is_file()
dyld = ctypes.CDLL(None)
dyld._dyld_image_count.restype = ctypes.c_uint32
dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
dyld._dyld_get_image_name.restype = ctypes.c_char_p
loaded = [dyld._dyld_get_image_name(index).decode() for index in range(dyld._dyld_image_count())]
external = [path for path in loaded if not path.startswith(('/System/Library/', '/usr/lib/')) and not Path(path).resolve().is_relative_to(root)]
assert not external, external
print(json.dumps({'prefix':sys.prefix, 'sysPath':sys.path, 'importedModules':paths,
                  'versions':{name:importlib.metadata.version(name) for name in
                      ['torch','torchvision','numpy','pillow','scipy','trimesh','xatlas','usd-core','opencv-python-headless','mlx','mlx-metal']},
                  'cpuSmokeChecksPassed':True, 'mpsAvailable':torch.backends.mps.is_available(),
                  'mlxCorePath':str(mlx_core), 'mlxExecution':'not tested',
                  'uvLayoutLibrarySHA256':layout_report['nativeLibrarySHA256'],
                  'loadedMachOImages':loaded, 'externalNonSystemLibraries':external,
                  'checks':['xatlas face mapping', 'PNG roundtrip', 'OpenCV colors', 'mesh normals', 'thin silhouette rim preservation', 'Torch CPU matrix', 'Torchvision NMS', 'OpenUSD mesh', 'packaged resource retry protocol', 'continuous UV overlap detection and repair', 'loaded native library paths']}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Report must be new')
    root = args.runtime.resolve()
    env = {'PATH':'/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE':'1', 'HF_HUB_OFFLINE':'1'}
    if 'TMPDIR' in os.environ:
        env['TMPDIR'] = os.environ['TMPDIR']
    command = [str(root/'python/bin/python3.12'), '-I', '-B', '-c', PROBE, str(root)]
    result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=120)
    report = {'scope':'Packaged CPU smoke checks; not Metal, application or full offline acceptance',
              'runtimeManifestSHA256':hashlib.sha256((root/'runtime.json').read_bytes()).hexdigest(),
              'probeScriptSHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'exitCode':result.returncode, 'stderr':result.stderr,
              'isolatedMode':True, 'developerEnvironmentInherited':False}
    if result.returncode == 0:
        report['probe'] = json.loads(result.stdout)
    else:
        report['stdout'] = result.stdout
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
