#!/usr/bin/env python3
"""Analytic fixtures for the offline layout auditor; no production model data."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('layout_audit', Path(__file__).with_name('audit-uv-layout.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--auditor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ('separated', [[1,1],[5,1],[1,5],[8,1],[12,1],[8,5]], [[0,1,2],[3,4,5]], 0, 0, 3),
        ('duplicate', [[1,1],[5,1],[1,5]]*2, [[0,1,2],[3,4,5]], 1, 8, 0),
        ('shared-edge', [[1,1],[5,1],[1,5],[5,5]], [[0,1,2],[1,3,2]], 0, 0, None),
        ('separate-touching', [[1,1],[5,1],[1,5],[5,1],[5,5],[1,5]], [[0,1,2],[3,4,5]], 0, 0, 0),
        ('same-island-fold', [[1,1],[5,1],[1,5],[3,1],[1,3]], [[0,1,2],[0,3,4]], 1, 2, None),
        ('grid-boundary', [[14,1],[18,1],[14,5],[21,1],[25,1],[21,5]], [[0,1,2],[3,4,5]], 0, 0, 3),
        ('reversed-winding', [[1,1],[5,1],[1,5]]*2, [[0,1,2],[5,4,3]], 1, 8, 0),
        ('overlapping-boxes-disjoint-triangles', [[1,1],[2,1],[1,2],[1.8,1.8],[2.8,1.8],[1.8,2.8]],
         [[0,1,2],[3,4,5]], 0, 0, .6/np.sqrt(2)),
    ]
    results = []
    for name, vertices, faces, overlaps, area, gap in cases:
        uv = args.output/(name+'.npz')
        np.savez(uv, uv=np.array(vertices, dtype='float64')/32,
                 faces=np.array(faces, dtype='int32'), width=32, height=32)
        report = audit.inspect(uv, args.output/name, args.auditor, 4)
        assert report['overlappingTrianglePairsAbove1eMinus8TexelSquared'] == overlaps, name
        assert abs(report['maximumIntersectionTexelSquared'] - area) < 1e-10, name
        actual = report['minimumCrossChartGapTexels']
        assert actual is None if gap is None else actual is not None and abs(actual-gap)<1e-10, name
        assert report['crossChartOverlapPairs'] == (0 if name=='same-island-fold' else overlaps), name
        results.append({'case':name,'passed':True})
    (args.output/'summary.json').write_text(json.dumps({
        'testSourceSHA256':audit.sha(__file__), 'auditorBinarySHA256':audit.sha(args.auditor),
        'checks':results, 'scope':'Analytic overlap/gap fixtures, not application acceptance'}, indent=2)+'\n')
    print(f'{len(results)} analytic overlap/gap fixtures passed')


if __name__ == '__main__':
    main()
