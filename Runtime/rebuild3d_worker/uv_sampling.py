"""Restore subpixel UV collapses without changing any 3D surface corner.

Generic geometry/texel criteria select faces; no object or photo preferences.
New isometric patches retain the atlas maximum dimension. Final overlap and
photographic quality checks remain the caller's responsibility.
"""
import time

import numpy as np

from .uv_layout import validate


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


def repair_sampling(data, surface, padding, anisotropy_limit=4., select_altitude=1., target_altitude=2.5):
    started = time.monotonic()
    uv, faces, width, height = validate(data)
    if (surface.shape != (len(faces), 3, 3) or padding < 1 or
            not 1 < anisotropy_limit or not 0 < select_altitude < target_altitude):
        raise ValueError('Invalid surface or sampling configuration')
    pixels = uv.astype('float64')*[width, height]
    metrics = triangle_metrics(surface, pixels[faces])
    old_altitude = altitude(pixels[faces])
    selected = (metrics['anisotropy'] > anisotropy_limit) & (old_altitude < select_altitude)
    if not selected.any():
        return data, {'repairedFaces': 0, 'texelScale': 1., 'stripHeight': 0,
                      'selectedFaceIndices': [], 'medianHealthyAreaDensity': None,
                      'densitySource': 'not-required', 'elapsedSeconds': time.monotonic()-started}
    healthy = (metrics['anisotropy'] <= anisotropy_limit) & (old_altitude >= select_altitude)
    # If the whole atlas is collapsed, only the explicit patch-altitude floor
    # determines density; do not infer a healthy density from collapsed mappings.
    density = float(np.median(metrics['areaDensity'][healthy])) if healthy.any() else 0.
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
        after = triangle_metrics(surface, after_pixels)
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
        'densitySource': 'healthy-face-median' if healthy.any() else 'patch-altitude-floor-only',
        'minimumRepairedAltitudeTexels': float(after_altitude[ids].min()),
        'maximumRepairedAnisotropy': float(after['anisotropy'][ids].max()),
        'allSurfaceCornersPreserved': True, 'remainingSelectedSlivers': int(remaining.sum()),
        'elapsedSeconds': time.monotonic()-started}

