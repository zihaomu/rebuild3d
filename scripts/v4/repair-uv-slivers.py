#!/usr/bin/env python3
"""Offline prototype: restore nearly collapsed UV triangles from their 3D shape.

Select by surface-to-UV anisotropy and final texel altitude, never object IDs.
Isometric patches use the median density of healthy faces and a minimum texel
altitude; a padded strip is packed inside the same maximum atlas dimension.
This does not change the surface or prove photographic/native quality. A fresh
bake, layout audit and native review are mandatory before production adoption.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from rebuild3d_worker.common import sha, write_json
from rebuild3d_worker.uv_layout import overlapping_faces, validate

spec = importlib.util.spec_from_file_location('uv_distortion', Path(__file__).with_name('audit-uv-distortion.py'))
distortion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distortion)


def altitude(pixels):
    a, b = pixels[:, 1]-pixels[:, 0], pixels[:, 2]-pixels[:, 0]
    twice_area = abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    longest = np.linalg.norm(pixels-np.roll(pixels, 1, axis=1), axis=2).max(1)
    return twice_area/longest


def isometric_triangle(triangle):
    triangle = np.asarray(triangle, dtype='float64')
    lengths = np.linalg.norm(np.roll(triangle, -1, axis=0)-triangle, axis=1)
    i = int(np.argmax(lengths))
    j, k = (i+1) % 3, (i+2) % 3
    length = lengths[i]
    if not np.isfinite(triangle).all() or length <= 0:
        raise ValueError('Invalid surface triangle')
    edge = triangle[j]-triangle[i]
    delta = triangle[k]-triangle[i]
    height = np.linalg.norm(np.cross(edge, delta))/length
    if height <= 0:
        raise ValueError('Degenerate surface triangle')
    result = np.zeros((3, 2))
    result[j] = [length, 0]
    result[k] = [np.dot(delta, edge)/length, height]
    return result, height


def repair(data, surface, padding, anisotropy_limit=4., select_altitude=1., target_altitude=2.5):
    uv, faces, width, height = validate(data)
    if (surface.shape != (len(faces), 3, 3) or padding < 1 or
            not 1 < anisotropy_limit or not 0 < select_altitude < target_altitude):
        raise ValueError('Invalid surface or sampling configuration')
    pixels = uv.astype('float64')*[width, height]
    metrics = distortion.triangle_metrics(surface, pixels[faces])
    old_altitude = altitude(pixels[faces])
    healthy = (metrics['anisotropy'] <= anisotropy_limit) & (old_altitude >= select_altitude)
    if not healthy.any():
        raise ValueError('No healthy UV faces from which to estimate texel density')
    density = float(np.median(metrics['areaDensity'][healthy]))
    selected = (metrics['anisotropy'] > anisotropy_limit) & (old_altitude < select_altitude)
    if not selected.any():
        return data, {'repairedFaces': 0, 'texelScale': 1., 'stripHeight': 0,
                      'selectedFaceIndices': [], 'medianHealthyAreaDensity': density}
    original_corners = data['vertexMap'][faces]
    # Include any additional slivers crossing the selection boundary after scaling.
    for iteration in range(4):
        ids = np.flatnonzero(selected)
        patches = []
        for face_id in ids:
            patch, base_altitude = isometric_triangle(surface[face_id])
            patch *= max(np.sqrt(density), target_altitude/base_altitude)
            patches.append(patch)
        cell_sizes = np.array([np.ceil(p.max(0)-p.min(0)).astype(int)+2*padding+2 for p in patches])
        order = np.argsort(-cell_sizes[:, 1], kind='stable')
        positions = np.zeros((len(ids), 2))
        x = y = row_height = 0
        for index in order:
            cell_width, cell_height = cell_sizes[index]
            if cell_width > width:
                raise ValueError('Isometric patch does not fit atlas; no silent rescaling/distortion')
            if x+cell_width > width:
                x, y, row_height = 0, y+row_height, 0
            positions[index] = [x+padding+1, y+padding+1]
            x += int(cell_width)
            row_height = max(row_height, int(cell_height))
        strip = y+row_height+padding+1
        factor = max(width, height)/max(width, height+strip)
        dimensions = np.array([round(width*factor), round((height+strip)*factor)])
        uv_faces = faces.copy()
        mapping = np.concatenate([data['vertexMap'], original_corners[ids].ravel()])
        uv_faces[ids] = np.arange(len(data['vertexMap']), len(mapping)).reshape(-1, 3)
        patch_pixels = np.concatenate([p+position for p, position in zip(patches, positions)])
        new_pixels = np.vstack([pixels+[0, strip], patch_pixels])
        new_uv = (new_pixels/[width, height+strip]).astype('float32')
        result = dict(data, uv=new_uv, faces=uv_faces, vertexMap=mapping,
                      width=np.array(dimensions[0]), height=np.array(dimensions[1]))
        after_pixels = new_uv.astype('float64')[uv_faces]*dimensions
        after = distortion.triangle_metrics(surface, after_pixels)
        after_altitude = altitude(after_pixels)
        remaining = (after['anisotropy'] > anisotropy_limit) & (after_altitude < select_altitude)
        if not remaining.any():
            break
        if np.any(remaining & selected):
            raise ValueError('Repaired patch did not resolve sampling collapse')
        selected |= remaining
    else:
        raise ValueError('Sampling selection did not converge')
    if not np.array_equal(mapping[uv_faces], original_corners):
        raise ValueError('UV repair changed original corner identities')
    return result, {'repairedFaces': len(ids), 'selectedFaceIndices': ids.tolist(),
        'texelScale': factor, 'stripHeight': strip, 'selectionIterations': iteration+1,
        'medianHealthyAreaDensity': density, 'surfaceAreaFraction': float(
            metrics['surfaceArea'][ids].sum()/metrics['surfaceArea'].sum()),
        'minimumRepairedAltitudeTexels': float(after_altitude[ids].min()),
        'maximumRepairedAnisotropy': float(after['anisotropy'][ids].max()),
        'allSurfaceCornersPreserved': True, 'remainingSelectedSlivers': int(remaining.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry', type=Path, help='bundle geometry.npz')
    parser.add_argument('uv_directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with np.load(args.geometry) as geometry, np.load(args.uv_directory/'uv.npz') as uv:
        data = dict(uv)
        original_ids = data['originalFaceID']
        if not np.array_equal(np.sort(original_ids), np.arange(len(geometry['faces']))):
            raise ValueError('Original face identities are incomplete')
        if not np.array_equal(data['vertexMap'][data['faces']], geometry['faces'][original_ids]):
            raise ValueError('Original face corner mapping differs')
        surface = geometry['vertices'][geometry['faces'][original_ids]]
    parent = json.loads((args.uv_directory/'report.json').read_text())
    padding = int(parent['paddingPixels'])
    old_affected, before = overlapping_faces(data)
    if len(old_affected):
        raise ValueError('This prototype expects overlap-repaired input')
    result, info = repair(data, surface, padding)
    affected, after = overlapping_faces(result)
    if len(affected):
        raise ValueError('Sampling repair introduced overlap')
    np.savez_compressed(args.output/'uv.npz', **result)
    report = {'status': 'offline-prototype-needs-fresh-bake-and-native-review',
        'policy': 'isometric-subpixel-slivers-v1', 'scriptSHA256': sha(__file__),
        'geometrySHA256': sha(args.geometry), 'parentUVSHA256': sha(args.uv_directory/'uv.npz'),
        'criteria': {'anisotropyAbove': 4., 'altitudeBelowTexels': 1., 'patchAltitudeBeforeScaling': 2.5},
        'criteriaPurpose': 'Generic algorithm parameters; not a replacement for original visual quality gates',
        'parent': parent, 'width': int(result['width']), 'height': int(result['height']),
        'paddingPixels': max(1, int(np.floor(padding*info['texelScale']))),
        'beforeOverlap': before, 'afterOverlap': after, 'sliverRepair': info,
        'repairedOriginalFaceIDs': original_ids[info['selectedFaceIndices']].tolist(),
        'outputUVSHA256': sha(args.output/'uv.npz'), 'requiresFreshPhotographicBake': True,
        'elapsedSeconds': time.monotonic()-start}
    write_json(args.output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('width', 'height', 'paddingPixels', 'sliverRepair', 'elapsedSeconds')}), flush=True)


if __name__ == '__main__':
    main()
