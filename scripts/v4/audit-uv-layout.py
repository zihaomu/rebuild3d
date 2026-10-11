#!/usr/bin/env python3
"""Check continuous atlas triangle overlap and distances between UV boundaries.

UV-connected components identify islands. Gaps are measured in final atlas
texels, not inferred from configured padding. This is not native renderer or
mipmap acceptance; tiny intersections below 1e-8 square texels are roundoff.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact_intersection_area(a, b):
    """Recheck reported examples using exact rationals of the stored UV values."""
    def sub(p, q):
        return tuple(x-y for x, y in zip(p, q))
    def cross(p, q):
        return p[0]*q[1]-p[1]*q[0]
    polygon = [tuple(Fraction(float(x)) for x in p) for p in a]
    clip = [tuple(Fraction(float(x)) for x in p) for p in b]
    if cross(sub(clip[1], clip[0]), sub(clip[2], clip[0])) < 0:
        clip[1], clip[2] = clip[2], clip[1]
    for e in range(3):
        result = []
        start, edge = clip[e], sub(clip[(e+1) % 3], clip[e])
        for i, p in enumerate(polygon):
            q = polygon[(i+1) % len(polygon)]
            dp, dq = cross(edge, sub(p, start)), cross(edge, sub(q, start))
            if dp >= 0:
                result.append(p)
            if (dp >= 0) != (dq >= 0):
                result.append(tuple(x+(y-x)*dp/(dp-dq) for x, y in zip(p, q)))
        polygon = result
    twice = sum((cross(sub(polygon[i], polygon[0]), sub(polygon[i+1], polygon[0]))
                 for i in range(1, len(polygon)-1)), Fraction(0))
    return float(abs(twice)/2)


def inspect(uv_path, output, executable, padding):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with np.load(uv_path) as source:
        uv = source['uv'].astype('float64')
        faces = source['faces'].astype('int32')
        width, height = int(source['width']), int(source['height'])
    if not np.isfinite(uv).all() or not ((uv >= 0) & (uv <= 1)).all():
        raise ValueError('UVs must be finite and inside the atlas')
    if faces.min() < 0 or faces.max() >= len(uv):
        raise ValueError('Invalid UV face indices')
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix((np.ones(len(edges), dtype='uint8'), (edges[:, 0], edges[:, 1])),
                       shape=(len(uv), len(uv))).tocsr()
    _, component = connected_components(graph, directed=False)
    face_components = component[faces]
    assert np.all(face_components == face_components[:, :1])
    chart_ids, face_labels = np.unique(face_components[:, 0], return_inverse=True)
    mapping = np.full(len(uv), -1, dtype='int32')
    mapping[chart_ids] = np.arange(len(chart_ids), dtype='int32')
    boundary = edges[counts == 1]
    boundary_labels = mapping[component[boundary[:, 0]]]
    assert (boundary_labels >= 0).all()
    pixels = uv * [width, height]
    triangles = np.ascontiguousarray(pixels[faces], dtype='<f8')
    segments = np.ascontiguousarray(pixels[boundary], dtype='<f8')
    files = [output/name for name in ('triangles.bin','face-labels.bin','boundaries.bin','boundary-labels.bin')]
    for path, value in zip(files, [triangles, face_labels.astype('<i4'), segments, boundary_labels.astype('<i4')]):
        value.tofile(path)
    radius = max(2 * padding + 2, 4)
    command = [str(executable.resolve()), *map(str, files), str(len(faces)), str(len(boundary)),
               str(len(chart_ids)), str(width), str(height), str(radius), str(output/'native.json')]
    with (output/'native.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Continuous layout audit failed; see {output / "native.log"}')
    report = json.loads((output/'native.json').read_text())
    pairs = np.array(report['exampleOverlappingFacePairsFlat'], dtype=int).reshape(-1, 2)
    exact_areas = [exact_intersection_area(triangles[a], triangles[b]) for a, b in pairs]
    if not all(area > 1e-8 for area in exact_areas):
        raise ValueError('A reported intersection failed exact-arithmetic recheck')
    gaps = np.array([np.inf if x is None else x for x in report.pop('chartMinimumGapTexels')])
    report.update({
        'scope': 'Continuous UV layout diagnostic; not final reconstruction/native rendering acceptance',
        'uvSHA256': sha(uv_path), 'scriptSHA256': sha(__file__), 'auditorBinarySHA256': sha(executable),
        'uvPath': str(uv_path.resolve()), 'atlasSize': [width, height], 'triangleCount': len(faces),
        'uvConnectedComponents': len(chart_ids), 'boundaryEdges': len(boundary),
        'nonManifoldUVEdges': int((counts > 2).sum()), 'configuredPaddingTexels': padding,
        'exampleIntersectionAreasExactRational': exact_areas,
        'reportedExamplesConfirmedByExactArithmetic': True,
        'chartsWithGapBelow1Texel': int((gaps < 1).sum()),
        'chartsWithGapBelow2Texels': int((gaps < 2).sum()),
        'chartsWithGapBelowTwiceConfiguredPadding': int((gaps < 2 * padding).sum()),
        'chartsWithNoNeighborInsideSearchRadius': int(np.isinf(gaps).sum()),
        'elapsedSeconds': time.monotonic()-started,
        'limits': 'Components use shared UV vertex indices. Same-component triangle folds are checked for overlap, but their internal boundary gaps are not classified as cross-island gaps. Missing nearest gaps are lower-bounded by the search radius, not zero. No claim about all mip levels or SceneKit sampling.'})
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uv', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--auditor', type=Path, required=True)
    parser.add_argument('--padding', type=int, required=True)
    args = parser.parse_args()
    if args.padding < 1:
        raise ValueError('Expected positive actual atlas padding')
    print(json.dumps(inspect(args.uv, args.output, args.auditor, args.padding), indent=2))


if __name__ == '__main__':
    main()
