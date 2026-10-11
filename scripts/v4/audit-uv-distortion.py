#!/usr/bin/env python3
"""Compare triangle stretch on one fixed surface before/after UV changes.

Singular values map an orthonormal surface basis to atlas texels. Their ratio
measures anisotropy (1 = no directional stretch), while their product measures
texels per surface area. Report face-count and surface-area weighted statistics;
these are diagnostics, not substitute thresholds for visual or native QA.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def triangle_metrics(surface, pixels):
    surface = np.asarray(surface, dtype='float64')
    pixels = np.asarray(pixels, dtype='float64')
    if (surface.ndim != 3 or surface.shape[1:] != (3, 3)
            or pixels.shape != (len(surface), 3, 2) or not len(surface)
            or not np.isfinite(surface).all() or not np.isfinite(pixels).all()):
        raise ValueError('Expected matching finite, nonempty 3D and UV triangles')
    a, b = surface[:, 1] - surface[:, 0], surface[:, 2] - surface[:, 0]
    length = np.linalg.norm(a, axis=1)
    twice_area = np.linalg.norm(np.cross(a, b), axis=1)
    if np.any(length <= 0) or np.any(twice_area <= 0):
        raise ValueError('Degenerate 3D triangle; no faces may be silently omitted')
    x = np.einsum('ij,ij->i', a, b) / length
    y = twice_area / length
    u, v = pixels[:, 1] - pixels[:, 0], pixels[:, 2] - pixels[:, 0]
    if np.any(u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0] == 0):
        raise ValueError('Degenerate UV triangle; no faces may be silently omitted')
    first = u / length[:, None]
    second = (v - first * x[:, None]) / y[:, None]
    jacobian = np.stack((first, second), axis=-1)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if not np.isfinite(singular).all() or np.any(singular[:, 1] <= 0):
        raise ValueError('Degenerate or nonfinite UV Jacobian')
    return {'surfaceArea': twice_area / 2,
            'anisotropy': singular[:, 0] / singular[:, 1],
            'areaDensity': singular[:, 0] * singular[:, 1]}


def quantiles(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    if (values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError('Quantiles require finite values and positive weights')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    result = {}
    for percentile in (0, 50, 95, 99, 100):
        index = min(int(np.searchsorted(cumulative, cumulative[-1]*percentile/100)), len(values)-1)
        result['p' + str(percentile)] = float(values[order[index]])
    return result


def summary(values, areas):
    return {'byFaceCount': quantiles(values, np.ones(len(values))),
            'bySurfaceArea': quantiles(values, areas)}


def load_uv(path, surface_faces):
    with np.load(path) as data:
        mapping, faces, ids = data['vertexMap'], data['faces'], data['originalFaceID']
        uv = data['uv'].astype('float64')
        size = np.array([int(data['width']), int(data['height'])])
    if (faces.shape != surface_faces.shape or ids.shape != (len(surface_faces),)
            or mapping.ndim != 1 or uv.shape != (len(mapping), 2)
            or any(x.dtype.kind not in 'iu' for x in (mapping, faces, ids))
            or np.any(size <= 0) or not np.isfinite(uv).all()
            or np.any(uv < 0) or np.any(uv > 1)):
        raise ValueError('Invalid atlas shapes, indices, bounds or dimensions')
    if not np.array_equal(np.sort(ids), np.arange(len(surface_faces))):
        raise ValueError('Every original face must occur exactly once')
    if faces.min() < 0 or faces.max() >= len(mapping):
        raise ValueError('UV face index out of bounds')
    if not np.array_equal(mapping[faces], surface_faces[ids]):
        raise ValueError('Original face/corner mapping changed')
    return (uv[faces] * size)[np.argsort(ids)], size.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry', type=Path)
    parser.add_argument('before_uv', type=Path)
    parser.add_argument('after_uv', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.geometry) as data:
        vertices, faces = data['verticesBeforeDisplay'], data['faces']
        completion = data['completion'].astype(bool)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Invalid source surface indices or dimensions')
    if completion.shape != (len(faces),):
        raise ValueError('Missing per-face provenance')
    surface = vertices[faces]
    metrics, sizes = {}, {}
    for label, path in [('before', args.before_uv), ('after', args.after_uv)]:
        pixels, sizes[label] = load_uv(path, faces)
        metrics[label] = triangle_metrics(surface, pixels)
    areas = metrics['before']['surfaceArea']
    assert np.array_equal(areas, metrics['after']['surfaceArea'])
    report = {'scope': __doc__, 'scriptSHA256': sha(__file__),
              'geometrySHA256': sha(args.geometry),
              'uvSHA256': {'before': sha(args.before_uv), 'after': sha(args.after_uv)},
              'atlasSize': sizes, 'triangleCount': len(faces),
              'allOriginalFaceCornersPreserved': True, 'invalidOrOmittedFaces': 0,
              'thresholdPolicy': 'Descriptive values only; no new visual acceptance thresholds',
              'groups': {}}
    arrays = {'originalFaceID': np.arange(len(faces)), 'completion': completion, 'surfaceArea': areas}
    for label, data in metrics.items():
        for name in ('anisotropy', 'areaDensity'):
            arrays[label + '_' + name] = data[name]
    for group, mask in [('all', np.ones(len(faces), dtype=bool)),
                        ('learnedDepth', ~completion), ('completion', completion)]:
        if not mask.any():
            report['groups'][group] = {'faces': 0}
            continue
        weights = areas[mask]
        item = {'faces': int(mask.sum()), 'surfaceAreaFraction': float(weights.sum()/areas.sum())}
        for label, data in metrics.items():
            mean_density = float(np.average(data['areaDensity'], weights=areas))
            anisotropy = data['anisotropy'][mask]
            item[label] = {
                'anisotropy': summary(anisotropy, weights),
                'normalizedLinearTexelDensity': summary(np.sqrt(data['areaDensity'][mask]/mean_density), weights),
                'diagnosticSurfaceAreaFractionAnisotropyAbove': {
                    str(limit): float(weights[anisotropy > limit].sum()/weights.sum()) for limit in (2, 4, 16, 1000)}}
        for name in ('anisotropy', 'areaDensity'):
            ratio = metrics['after'][name][mask] / metrics['before'][name][mask]
            item['afterToBefore_' + name] = summary(ratio, weights)
        report['groups'][group] = item
    ratio = metrics['after']['anisotropy']/metrics['before']['anisotropy']
    ids = np.argsort(ratio)[-20:][::-1]
    report['largestAnisotropyIncreaseFaces'] = [
        {'originalFaceID': int(i), 'completion': bool(completion[i]),
         'surfaceAreaFraction': float(areas[i]/areas.sum()),
         'before': float(metrics['before']['anisotropy'][i]),
         'after': float(metrics['after']['anisotropy'][i]), 'ratio': float(ratio[i])} for i in ids]
    np.savez_compressed(args.output/'face-metrics.npz', **arrays)
    report['perFaceMetricsSHA256'] = sha(args.output/'face-metrics.npz')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'triangleCount':len(faces), 'all':report['groups']['all']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
