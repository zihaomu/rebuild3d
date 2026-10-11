#!/usr/bin/env python3
"""Prototype: move every audited overlapping face to its own padded UV island.

Only UV corners change. Original 3D vertices, face order and source IDs remain
identical. This is an offline experiment; adoption requires a fresh bake,
quality review and production integration. Never reuses previous texture colors.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('layout_audit', Path(__file__).with_name('audit-uv-layout.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('uv_directory', type=Path)
    p.add_argument('audit_report', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--auditor', type=Path, required=True)
    p.add_argument('--size', type=int, default=4096)
    args = p.parse_args()
    prior = json.loads((args.uv_directory/'report.json').read_text())
    layout = json.loads(args.audit_report.read_text())
    uv_path = args.uv_directory/'uv.npz'
    if layout['uvSHA256'] != audit.sha(uv_path):
        raise ValueError('Audit does not describe these exact UVs')
    affected = np.array(layout['affectedFaceIDs'], dtype='int64')
    if len(affected) != layout['affectedFaces'] or len(np.unique(affected)) != len(affected) or not len(affected):
        raise ValueError('Expected a complete, unique, nonempty affected face list')
    data = dict(np.load(uv_path))
    uv, faces, mapping = data['uv'], data['faces'].copy(), data['vertexMap'].copy()
    if affected.min() < 0 or affected.max() >= len(faces):
        raise ValueError('Invalid affected face ID')
    original_corners = mapping[faces].copy()
    width, height = int(data['width']), int(data['height'])
    padding = int(prior['paddingPixels'])
    pixels = uv.astype('float64')*[width, height]
    patches = []
    x = y = row_height = 0
    for face_id in affected:
        triangle = pixels[faces[face_id]]
        local = triangle-triangle.min(axis=0)
        span = local.max(axis=0)
        cell_width, cell_height = np.ceil(span).astype(int)+2*padding+2
        if cell_width > width:
            raise ValueError('A triangle does not fit an isolated row; no implicit distortion is allowed')
        if x+cell_width > width:
            x, y, row_height = 0, y+row_height, 0
        patches.append(local+[x+padding+1, y+padding+1])
        x += int(cell_width)
        row_height = max(row_height, int(cell_height))
    strip_height = y+row_height+padding+1
    pixels[:, 1] += strip_height
    start = len(mapping)
    mapping = np.concatenate([mapping, original_corners[affected].ravel()])
    faces[affected] = np.arange(start, start+3*len(affected)).reshape(-1, 3)
    combined = np.vstack([pixels, np.concatenate(patches)])/[width, height+strip_height]
    factor = args.size/max(width, height+strip_height)
    final_width, final_height = round(width*factor), round((height+strip_height)*factor)
    data.update(uv=combined.astype('float32'), faces=faces, vertexMap=mapping,
                width=np.array(final_width), height=np.array(final_height))
    assert np.array_equal(mapping[faces], original_corners)
    triangle_uv = data['uv'].astype('float64')[faces]
    ab, ac = triangle_uv[:, 1]-triangle_uv[:, 0], triangle_uv[:, 2]-triangle_uv[:, 0]
    area = abs(ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0])/2
    if not (area > 0).all():
        raise ValueError('Repair introduced a degenerate UV triangle')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'uv.npz', **data)
    final_padding = max(1, math.floor(padding*factor))
    result = audit.inspect(args.output/'uv.npz', args.output/'layout-audit', args.auditor, final_padding)
    report = {
        'scope': 'UV overlap-repair prototype; fresh bake and full quality acceptance pending',
        'scriptSHA256': audit.sha(__file__), 'parentUVSHA256': audit.sha(uv_path),
        'inputAuditSHA256': audit.sha(args.audit_report), 'parent': prior,
        'width': final_width, 'height': final_height, 'paddingPixels': final_padding,
        'originalFaceCornerOrderPreserved': True, 'sameOriginalFaceIDs': True,
        'repairedOverlappingFaceIDs': affected.tolist(), 'repairedOverlappingFaces': len(affected),
        'addedStripHeightBeforeScaling': strip_height, 'texelScaleAfterAddingIslands': factor,
        'triangleShapesPreservedBeforeFinalAtlasRounding': True,
        'minimumUVTriangleArea': float(area.min()), 'requiresFreshPhotographicBake': True,
        'remainingOverlapPairs': result['overlappingTrianglePairsAbove1eMinus8TexelSquared'],
        'finalNativeApplicationAcceptance': 'pending'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if report['remainingOverlapPairs']:
        raise ValueError('Repair did not eliminate all measured overlap; retain candidate as unaccepted')
    print(json.dumps({k:v for k,v in report.items() if k not in ('parent', 'repairedOverlappingFaceIDs')}, indent=2))


if __name__ == '__main__':
    main()
