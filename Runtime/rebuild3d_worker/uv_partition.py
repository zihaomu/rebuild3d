"""Bound xatlas chart construction without changing the original surface."""
import time

import numpy as np
import xatlas


DEFAULT_FACE_BUDGET = 32768


def partition_faces(vertices, faces, budget):
    if budget < 0:
        raise ValueError('UV face budget cannot be negative')
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('UV input must be a finite, nonempty surface')
    ids = np.arange(len(faces), dtype=np.int64)
    if not budget:
        return [ids]
    centers = vertices[faces].mean(axis=1)
    pending, groups = [ids], []
    while pending:
        current = pending.pop()
        if len(current) <= budget:
            groups.append(np.sort(current))
            continue
        axis = np.ptp(centers[current], axis=0).argmax()
        order = current[np.argsort(centers[current, axis], kind='stable')]
        split = len(order) // 2
        pending.extend((order[split:], order[:split]))
    if not np.array_equal(np.sort(np.concatenate(groups)), ids):
        raise ValueError('UV partition lost or duplicated original faces')
    return groups


def generate_uv(vertices, faces, normals, resolution, budget):
    start = time.monotonic()
    groups = partition_faces(vertices, faces, budget)
    atlas = xatlas.Atlas()
    remaps = []
    for ids in groups:
        used, inverse = np.unique(faces[ids], return_inverse=True)
        # Keep the original whole-surface normals at every partition boundary.
        atlas.add_mesh(vertices[used], inverse.reshape(-1, 3).astype('uint32'), normals[used])
        remaps.append(used)
    chart, packing = xatlas.ChartOptions(), xatlas.PackOptions()
    packing.resolution = resolution
    packing.padding = 4
    packing.bilinear = True
    setup = time.monotonic() - start
    generated = time.monotonic()
    if hasattr(atlas, 'generate_profile'):
        phases = atlas.generate_profile(chart, packing)
    else:
        atlas.generate(chart_options=chart, pack_options=packing)
        phases = {'computeChartsSeconds': None, 'packChartsSeconds': None}
    phases.update(partitionAndAddMeshSeconds=setup, generateSeconds=time.monotonic()-generated)
    if atlas.atlas_count != 1:
        raise ValueError('Expected one UV atlas; refusing to drop additional atlases')
    maps, uvs, offset = [], [], 0
    indices = np.empty(faces.shape, dtype='uint32')
    for index, ids in enumerate(groups):
        mapping, local_faces, uv = atlas[index]
        maps.append(remaps[index][mapping])
        uvs.append(uv)
        indices[ids] = local_faces + offset
        offset += len(mapping)
    mapping, uv = np.concatenate(maps), np.concatenate(uvs)
    if not np.array_equal(mapping[indices], faces):
        raise ValueError('UV generation changed original face/corner order')
    return {'vertexMap': mapping, 'faces': indices, 'uv': uv,
            'width': atlas.width, 'height': atlas.height,
            'charts': sum(atlas.get_mesh_chart_count(i) for i in range(len(groups))),
            'utilization': atlas.utilization, 'chunkFaceBudget': budget, 'meshChunks': len(groups)}, phases
