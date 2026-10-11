#!/usr/bin/env python3
"""Check unlit dark pixels against fused and exported triangles independently.

The camera matrices must be captured from the actual fixed SceneKit view.
The ROI is a post-review diagnostic, never a replacement quality threshold.
No geometry is changed and this probe does not establish visual acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom
from scipy.ndimage import binary_erosion, binary_fill_holes, label


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ray_hits(triangles, origin, direction):
    """Double-sided Moller-Trumbore in float64, independent of the rasterizer."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(direction, edge2)
    determinant = np.einsum('ij,ij->i', edge1, cross)
    valid = np.abs(determinant) > 1e-14
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1 / determinant[valid]
    delta = origin - triangles[:, 0]
    u = np.einsum('ij,ij->i', delta, cross) * inverse
    q = np.cross(delta, edge1)
    v = (q @ direction) * inverse
    distance = np.einsum('ij,ij->i', edge2, q) * inverse
    valid &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (distance > 1e-6)
    indices = np.flatnonzero(valid)
    order = np.argsort(distance[indices])
    indices = indices[order]
    return [{'triangle': int(i), 'distance': float(distance[i]),
             'pointDisplay': (origin + direction * distance[i]).tolist()} for i in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('render', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--other-mask', type=Path, required=True)
    parser.add_argument('--roi', nargs=4, type=int, required=True, metavar=('X0', 'Y0', 'X1', 'Y1'))
    args = parser.parse_args()
    assert not args.output.exists()
    camera_file = args.render / 'camera.json'
    camera = json.loads(camera_file.read_text())
    width, height = camera['imageSize']
    mask_file = args.render / 'turntable-mask-08.png'
    mask = np.asarray(Image.open(mask_file).convert('L'))
    other = np.asarray(Image.open(args.other_mask).convert('L'))
    assert mask.shape == other.shape == (height, width)
    interior = binary_erosion(binary_fill_holes(mask >= 128) & binary_fill_holes(other >= 128), iterations=8)
    x0, y0, x1, y1 = args.roi
    assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
    region = np.zeros_like(interior)
    region[y0:y1, x0:x1] = True
    components, count = label((mask < 64) & interior & region)
    source = args.job / 'fusion/face-provenance.npz'
    with np.load(source) as data:
        vertices = data['verticesBeforeDisplay'].astype(np.float64)
        faces = data['faces']
        completion = data['completion']
    fusion_report = args.job / 'fusion/report.json'
    display = np.asarray(json.loads(fusion_report.read_text())['worldToDisplay'], dtype=np.float64)
    if not np.allclose(display, camera['worldToDisplay'], rtol=0, atol=2e-6):
        raise ValueError('Captured renderer transform differs from the source fusion transform')
    vertices = vertices @ display[:3, :3].T + display[:3, 3]
    fused = vertices[faces]
    stage = Usd.Stage.Open(str(args.job / 'bundle/model.usdz'))
    xforms = UsdGeom.XformCache()
    exported = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        assert np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        # USD/Gf uses row-vector transforms; exported root transforms are
        # ordinarily identity, but do not assume that in the independent test.
        transform = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=np.float64)
        points = np.c_[points, np.ones(len(points))] @ transform
        points = points[:, :3] / points[:, 3, None]
        exported.append(points[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)])
    exported = np.concatenate(exported)
    assert len(exported) == len(fused)
    camera_to_world = np.asarray(camera['cameraToWorld'], dtype=np.float64)
    inverse_projection = np.linalg.inv(np.asarray(camera['projection'], dtype=np.float64))
    origin = camera_to_world[:3, 3]
    samples = []
    for component in range(1, count + 1):
        pixels = np.argwhere(components == component)
        center = pixels.mean(axis=0)
        y, x = pixels[np.argmin(np.linalg.norm(pixels - center, axis=1))]
        controls = [(x + dx, y + dy) for dx, dy in [(6, 0), (-6, 0), (0, 6), (0, -6)]
                    if 0 <= x + dx < width and 0 <= y + dy < height and mask[y + dy, x + dx] >= 240]
        assert controls, 'No adjacent opaque control pixel'
        for kind, px, py in [('dark-center', x, y), ('opaque-control', *controls[0])]:
            # Test center plus subpixel samples; no dependence on Metal's MSAA
            # sample pattern is needed to determine an open ray neighborhood.
            for ox, oy in [(.5, .5), (.25, .25), (.75, .25), (.25, .75), (.75, .75)]:
                clip = np.array([(px + ox) / width * 2 - 1, 1 - (py + oy) / height * 2, -1, 1])
                local = inverse_projection @ clip
                local = local[:3] / local[3]
                direction = camera_to_world[:3, :3] @ local
                direction /= np.linalg.norm(direction)
                fused_hits = ray_hits(fused, origin, direction)
                exported_hits = ray_hits(exported, origin, direction)
                for hit in fused_hits:
                    hit['completion'] = bool(completion[hit['triangle']])
                samples.append({'component': component, 'componentPixels': len(pixels), 'kind': kind,
                                'pixel': [int(px), int(py)], 'subpixel': [ox, oy], 'maskValue': int(mask[py, px]),
                                'fusedHits': fused_hits, 'exportedHits': exported_hits})
    controls = [s for s in samples if s['kind'] == 'opaque-control']
    if not controls or not all(s['fusedHits'] and s['exportedHits'] for s in controls):
        raise ValueError('Camera/ray mapping failed neighboring opaque control validation')
    dark = [s for s in samples if s['kind'] == 'dark-center']
    report = {'scope': __doc__, 'job': str(args.job), 'roiXYXY': args.roi,
              'cameraSHA256': sha(camera_file), 'maskSHA256': sha(mask_file), 'fusionSHA256': sha(source),
              'fusionReportSHA256': sha(fusion_report),
              'modelSHA256': sha(args.job / 'bundle/model.usdz'), 'scriptSHA256': sha(Path(__file__)),
              'darkComponents': count, 'darkSampleRays': len(dark),
              'darkRaysMissingFused': sum(not s['fusedHits'] for s in dark),
              'darkRaysMissingExported': sum(not s['exportedHits'] for s in dark),
              'allOpaqueControlsHitBoth': True, 'samples': samples}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['scope', 'samples']}))


if __name__ == '__main__':
    main()
