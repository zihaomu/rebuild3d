#!/usr/bin/env python3
"""Inspect depth and silhouette contributions between two mesh components.

The nearest-point segment is a diagnostic probe, never a proposed connector.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from rebuild3d_worker.geometry import project
from rebuild3d_worker.fuse import sample


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inference', type=Path)
    parser.add_argument('fusion', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--component', type=int, default=1)
    args = parser.parse_args()
    assert not args.output.exists()
    with np.load(args.fusion / 'face-provenance.npz') as geometry:
        vertices, faces = geometry['verticesBeforeDisplay'], geometry['faces']
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = sorted(trimesh.graph.connected_components(mesh.face_adjacency,
        nodes=np.arange(len(faces)), min_len=1), key=len, reverse=True)
    assert 0 < args.component < len(components)
    main = np.unique(faces[components[0]])
    part = np.unique(faces[components[args.component]])
    distance, indices = cKDTree(vertices[main]).query(vertices[part])
    index = int(np.argmin(distance))
    start, end = vertices[part[index]], vertices[main[indices[index]]]
    points = start[None] + (end-start)[None] * np.linspace(0, 1, 33)[:, None]
    fusion = json.loads((args.fusion / 'report.json').read_text())
    transforms = json.loads((args.inference / 'pixel-transforms.json').read_text())
    truncation = fusion['truncationWorld']
    weighted = np.zeros(len(points))
    weights = np.zeros(len(points))
    foreground = np.zeros(len(points), dtype=int)
    hull = np.full(len(points), -truncation)
    near_sum = np.zeros(len(points))
    near_weight = np.zeros(len(points))
    near_views = np.zeros(len(points), dtype=int)
    views = []
    with np.load(args.inference / 'predictions.npz') as data:
        for i, transform in enumerate(transforms):
            xy, z = project(points, data['extrinsics'][i], data['intrinsics'][i])
            left, top = transform['paddingLeftTop']
            width, height = transform['resizedSize']
            seen = ((z > 0) & (xy[:, 0] >= left) & (xy[:, 0] < left+width-1)
                    & (xy[:, 1] >= top) & (xy[:, 1] < top+height-1))
            binary = data['masks'][i].astype('uint8')
            inside = sample(binary, xy, order=0) > 0
            signed = (cv2.distanceTransform(1-binary, cv2.DIST_L2, 5)
                      - cv2.distanceTransform(binary, cv2.DIST_L2, 5))
            silhouette = ((sample(signed, xy, outside=1e3) - fusion['silhouetteTolerancePixels'])
                          * z / np.mean([data['intrinsics'][i, 0, 0], data['intrinsics'][i, 1, 1]]))
            hull = np.maximum(hull, np.where(seen, silhouette, -truncation))
            difference = sample(data['depth'][i], xy) - z
            use = seen & inside & (difference >= -truncation)
            certainty = np.clip(sample(data['confidence'][i], xy)-1, .1, 5.) * use
            weighted += np.clip(difference, -truncation, truncation) * certainty
            weights += certainty
            foreground += (seen & inside).astype(int)
            near = use & (np.abs(difference) <= truncation)
            near_sum += difference * certainty * near
            near_weight += certainty * near
            near_views += near.astype(int)
            views.append({'photoID': transform['photoID'], 'name': transform['name'],
                          'pixels': xy.tolist(), 'seen': seen.tolist(), 'insideMask': inside.tolist(),
                          'differenceOverTruncation': (difference/truncation).tolist(),
                          'certainty': certainty.tolist(), 'hullOverTruncation': (silhouette/truncation).tolist()})
    interior = np.where(foreground >= fusion['boundsRule']['interiorVotes'], -truncation, truncation)
    tsdf = np.where(weights > 0, weighted / np.maximum(weights, 1e-6), interior)
    near_field = np.where(near_views >= 2, near_sum / np.maximum(near_weight, 1e-6), tsdf)
    report = {'scope': __doc__, 'atUTC': datetime.now(timezone.utc).isoformat(),
              'status': 'diagnostic-not-a-geometry-edit', 'scriptSHA256': sha(Path(__file__)),
              'inferenceSHA256': sha(args.inference / 'predictions.npz'),
              'fusionSHA256': sha(args.fusion / 'face-provenance.npz'),
              'componentIndex': args.component, 'componentTriangles': len(components[args.component]),
              'gapWorld': float(distance[index]), 'gapVoxels': float(distance[index]/fusion['voxelSpacingWorld']),
              'sampledPoints': points.tolist(), 'views': views,
              'originalFieldOverTruncation': (np.maximum(tsdf, hull)/truncation).tolist(),
              'hullOverTruncation': (hull/truncation).tolist(), 'nearSurfaceViews': near_views.tolist(),
              'twoViewNearSurfaceHypothesisOverTruncation': (np.maximum(near_field, hull)/truncation).tolist(),
              'limits': 'Nearest segment is diagnostic only; it is not an observed centerline and must never be inserted into geometry.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'componentTriangles', 'gapWorld', 'gapVoxels')}))


if __name__ == '__main__':
    main()
