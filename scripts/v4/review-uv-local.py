#!/usr/bin/env python3
"""Locate selected UV faces and render identical surfaces with old/new atlases.

Run with the packaged Python and its worker on PYTHONPATH. Face-center rays
select an unoccluded recorded camera, or an explicitly synthetic diagnostic
camera when the recorded views cannot see that center. Native renders, CPU
visibility masks and contact sheets are evidence for visual review, not an
automatic quality pass. No source bundles are modified.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom

from rebuild3d_worker.common import camera_project, raster, sha, write_json


def first_hit(triangles, origin, target):
    direction = target - origin
    direction /= np.linalg.norm(direction)
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, e2)
    determinant = np.einsum('ij,ij->i', e1, h)
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant),
                        where=np.abs(determinant) > 1e-14)
    offset = origin - triangles[:, 0]
    u = np.einsum('ij,ij->i', offset, h) * inverse
    q = np.cross(offset, e1)
    v = (q @ direction) * inverse
    t = np.einsum('ij,ij->i', e2, q) * inverse
    hit = np.flatnonzero((np.abs(determinant) > 1e-14) & (u >= -1e-9)
                        & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-7))
    return int(hit[np.argmin(t[hit])]) if len(hit) else None


def choose_camera(vertices, faces, face_id, cameras, side=512, zoom=4):
    triangles = vertices[faces]
    triangle = triangles[face_id]
    center = triangle.mean(0)
    observations = []
    for index, record in enumerate(cameras):
        ext, k = np.asarray(record['worldToCamera']), np.asarray(record['workingIntrinsics'])
        origin = np.linalg.solve(ext[:, :3], -ext[:, 3])
        projected = camera_project(np.vstack((triangle, center)), ext, k).astype('float64')
        p = projected[:3, :2]
        area = abs(float(np.linalg.det(np.stack((p[1]-p[0], p[2]-p[0]))))) / 2
        width, height = record['workingSize']
        x, y, z = projected[3]
        nearest = first_hit(triangles, origin, center)
        observations.append({'cameraIndex': index, 'photoName': record['name'],
                             'centerPixel': [float(x), float(y)],
                             'projectedTriangleAreaPixels': area, 'firstHitFaceID': nearest,
                             'centerInFrame': bool(z > 0 and 0 <= x < width and 0 <= y < height),
                             'centerUnoccluded': nearest == face_id})
    visible = [o for o in observations if o['centerInFrame'] and o['centerUnoccluded']]
    if visible:
        selected = max(visible, key=lambda o: o['projectedTriangleAreaPixels'])
        record = copy.deepcopy(cameras[selected['cameraIndex']])
        center_pixel = np.asarray(selected['centerPixel'])
        transform = np.array([[zoom, 0, (side-1)/2-zoom*center_pixel[0]],
                              [0, zoom, (side-1)/2-zoom*center_pixel[1]], [0, 0, 1.]])
        record['workingIntrinsics'] = (transform @ np.asarray(record['workingIntrinsics'])).tolist()
        selection = {'kind': 'recorded-photo-camera-crop', 'cameraIndex': selected['cameraIndex'],
                     'sourcePhoto': record['name'], 'workingPixelZoom': zoom,
                     'workingToDiagnosticPixels': transform.tolist()}
    else:
        # The camera is solely for seeing the mesh surface, not a new observation.
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        normal /= np.linalg.norm(normal)
        edge = float(np.linalg.norm(triangle-np.roll(triangle, 1, axis=0), axis=1).max())
        found = None
        for multiple in (12, 6, 3, 1):
            for sign in (1, -1):
                origin = center + normal * sign * edge * multiple
                if first_hit(triangles, origin, center) == face_id:
                    found = (origin, multiple, sign)
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError(f'Cannot find an unoccluded diagnostic ray for face {face_id}')
        origin, multiple, sign = found
        forward = center-origin
        forward /= np.linalg.norm(forward)
        up = np.array([0., 1., 0.]) if abs(forward[1]) < .9 else np.array([1., 0., 0.])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        ext = np.c_[rotation, -rotation @ origin]
        k = np.array([[side, 0, (side-1)/2], [0, side, (side-1)/2], [0, 0, 1.]])
        record = {'worldToCamera': ext.tolist(), 'workingIntrinsics': k.tolist()}
        selection = {'kind': 'synthetic-surface-diagnostic', 'sourcePhoto': None,
                     'reason': 'No recorded camera has an in-frame unoccluded face center',
                     'distanceInTriangleLongestEdges': multiple, 'normalSign': sign,
                     'notPhotographicEvidence': True}
    record.update(name=f'face-{face_id}-{selection["kind"]}', workingSize=[side, side])
    return record, {'originalFaceID': face_id, 'recordedViewCenterRays': observations,
                    'selection': selection, 'camera': record}


def command_run(command, output, phase, results):
    start = time.monotonic()
    with (output/(phase+'.log')).open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    row = {'phase': phase, 'command': command, 'exitCode': result.returncode,
           'seconds': time.monotonic()-start}
    results.append(row)
    write_json(output/'process-results.json', results)
    print(json.dumps(row), flush=True)
    if result.returncode:
        raise RuntimeError(f'{phase} exited {result.returncode}; see saved log')


def tile(sheet, image, x, y, width, height, label):
    image = image.convert('RGB').copy()
    image.thumbnail((width-8, height-30), Image.Resampling.LANCZOS)
    sheet.paste(image, (x+(width-image.width)//2, y+27))
    ImageDraw.Draw(sheet).text((x+5, y+5), label, fill='white')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--faces', type=int, nargs='+', required=True)
    parser.add_argument('--renderer', type=Path, required=True)
    args = parser.parse_args()
    args.before, args.after, args.output, args.renderer = [p.resolve() for p in
        (args.before, args.after, args.output, args.renderer)]
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.before/'geometry.npz') as old, np.load(args.after/'geometry.npz') as new:
        for key in ('vertices', 'faces', 'normals', 'completion'):
            if not np.array_equal(old[key], new[key]):
                raise ValueError('Surface changed: '+key)
        vertices, faces, completion = new['vertices'], new['faces'], new['completion']
    camera_docs = [json.loads((p/'cameras.json').read_text()) for p in (args.before, args.after)]
    if camera_docs[0] != camera_docs[1]:
        raise ValueError('Cameras changed')
    if json.loads((args.before/'fusion.json').read_text())['worldToDisplay'] != json.loads(
            (args.after/'fusion.json').read_text())['worldToDisplay']:
        raise ValueError('Display transform changed')
    source_hashes, sampling = {}, {}
    for label, root in [('before', args.before), ('after', args.after)]:
        source_hashes[label] = {name: sha(root/name) for name in
                               ('geometry.npz', 'uv.npz', 'basecolor.png', 'cameras.json',
                                'fusion.json', 'atlas-surface.npz')}
        with np.load(root/'uv.npz') as uv, np.load(root/'atlas-surface.npz') as atlas:
            ids = uv['originalFaceID']
            if not np.array_equal(np.sort(ids), np.arange(len(faces))) or not np.array_equal(
                    uv['vertexMap'][uv['faces']], faces[ids]):
                raise ValueError('Original face/corner map changed')
            pixels = uv['uv'].astype('float64')[uv['faces']]*[int(uv['width']), int(uv['height'])]
            ab, ac = pixels[:, 1]-pixels[:, 0], pixels[:, 2]-pixels[:, 0]
            area = abs(ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0])/2
            altitude = 2*area/np.linalg.norm(pixels-np.roll(pixels, 1, axis=1), axis=2).max(1)
            raster_ids = atlas['originalFaceID']
            counts = np.bincount(raster_ids[raster_ids >= 0], minlength=len(faces))
            sampling[label] = {}
            for face_id in args.faces:
                index = int(np.flatnonzero(ids == face_id)[0])
                sampling[label][face_id] = {'texelArea': float(area[index]),
                    'minimumAltitudeTexels': float(altitude[index]),
                    'coveredAtlasTexelCenters': int(counts[face_id])}
    cameras = copy.deepcopy(camera_docs[1]['cameras'])
    photo_count = len(cameras)
    targets = []
    for face_id in args.faces:
        record, target = choose_camera(vertices, faces, face_id, cameras[:photo_count])
        target.update(completion=bool(completion[face_id]), renderViewIndex=len(cameras))
        target['atlasSampling'] = {label: sampling[label][face_id] for label in ('before', 'after')}
        cameras.append(record)
        _, hit_ids, _ = raster(camera_project(vertices, np.asarray(record['worldToCamera']),
                                             np.asarray(record['workingIntrinsics'])), faces, 512, 512)
        visible = hit_ids == face_id
        if not visible.any():
            raise ValueError(f'No visible raster samples for selected face {face_id}')
        target['diagnosticRasterVisibleSamples'] = int(visible.sum())
        target['visibilityMask'] = f'face-{face_id}-visible-mask.png'
        Image.fromarray(visible.astype('uint8')*255).save(args.output/target['visibilityMask'])
        targets.append(target)
    camera_path = args.output/'cameras.json'
    write_json(camera_path, {'cameras': cameras})
    report = {'scriptSHA256': sha(__file__), 'sourceBundles': {'before': str(args.before), 'after': str(args.after)},
              'sourceSHA256': source_hashes, 'rendererSHA256': sha(args.renderer),
              'cameraSHA256': sha(camera_path), 'sameGeometryNormalsProvenance': True,
              'originalPhotoViews': photo_count, 'targets': targets,
              'limits': 'CPU masks locate faces; visual acceptance requires actual native images. Center rays do not prove whole-face visibility. Synthetic views are not photographs.',
              'visualVerdict': 'pending'}
    write_json(args.output/'report.json', report)
    # Reexport only the old atlas with the current fixed exporter, preserving source files.
    reference = args.output/'before-bundle'
    reference.mkdir()
    for name in ('geometry.npz', 'uv.npz', 'fusion.json', 'basecolor.png'):
        (reference/name).symlink_to(args.before/name)
    results = []
    command_run([sys.executable, '-B', '-m', 'rebuild3d_worker.export', str(reference)],
                args.output, 'export-reference', results)
    for label, bundle in [('before', reference), ('after', args.after)]:
        renders = args.output/(label+'-renders')
        model = bundle/'model.usdz'
        command_run([str(args.renderer), str(model), str(renders), str(camera_path),
                     str(args.after/'fusion.json')], args.output, 'render-'+label, results)
        images = sorted(renders.glob('*.png'))
        if len(images) != 6*len(cameras)+36:
            raise ValueError('Native renderer did not produce all images')
        write_json(renders/'render-manifest.json', {
            'renderer': 'SceneKit native GPU offscreen; not application UI',
            'rendererSHA256': sha(args.renderer), 'modelSHA256': sha(model),
            'cameraSHA256': sha(camera_path), 'images': {p.name: sha(p) for p in images}})
    # Verify exported corners independently of the renderer and UV vertex splitting.
    stages = [Usd.Stage.Open(str(p/'model.usdz')) for p in (reference, args.after)]
    exported_corners = {}
    for region in ('region_0', 'region_1'):
        prims = [stage.GetPrimAtPath('/Object/'+region) for stage in stages]
        if not any(bool(p) for p in prims):
            continue
        if not all(bool(p) for p in prims):
            raise ValueError('Exported source groups changed')
        meshes = [UsdGeom.Mesh(p) for p in prims]
        indices = [np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3) for m in meshes]
        exported_corners[region] = {}
        for attribute in ('GetPointsAttr', 'GetNormalsAttr'):
            corners = [np.asarray(getattr(m, attribute)().Get())[i] for m, i in zip(meshes, indices)]
            same = bool(np.array_equal(*corners))
            exported_corners[region][attribute] = same
            if not same:
                raise ValueError('Exported positions/normals changed: '+region+' '+attribute)
    report['exportedCornerComparisons'] = exported_corners
    # Retain image differences even when exported positions and normals are equal.
    same_surface, difference_stats = {}, {}
    for p in (args.output/'before-renders').glob('*.png'):
        if '-clay' in p.name or '-mask' in p.name:
            a = np.asarray(Image.open(p)).astype('int16')
            b = np.asarray(Image.open(args.output/'after-renders'/p.name)).astype('int16')
            delta = abs(a-b)
            same_surface[p.name] = bool(not np.any(delta))
            difference_stats[p.name] = {'changedPixels': int(np.any(delta > 0, axis=-1).sum()),
                'maximumChannelDifference': int(delta.max()), 'meanChannelDifference': float(delta.mean())}
    report['sameSurfaceImageComparisons'] = same_surface
    report['surfaceImageDifferenceStatistics'] = difference_stats
    report['allClayAndMaskImagesIdentical'] = all(same_surface.values())
    sheet = Image.new('RGB', (1536, len(targets)*546), '#222222')
    for row, target in enumerate(targets):
        index, face_id = target['renderViewIndex'], target['originalFaceID']
        mask = np.asarray(Image.open(args.output/target['visibilityMask'])) > 0
        for column, label in enumerate(('before', 'after')):
            image = Image.open(args.output/(label+'-renders')/f'view-{index}-color.png')
            tile(sheet, image, column*512, row*546, 512, 546, f'{face_id} {label}; {target["selection"]["kind"]}')
        image = np.asarray(Image.open(args.output/'after-renders'/f'view-{index}-color.png').convert('RGB')).copy()
        image[mask] = np.array([255, 0, 255], dtype='uint8')
        tile(sheet, Image.fromarray(image), 1024, row*546, 512, 546, 'Diagnostic visible face in magenta')
    sheet.save(args.output/'local-faces.jpg', quality=96)
    for size in (400, 200):
        sheet = Image.new('RGB', (photo_count*400, 2*430), '#222222')
        for column in range(photo_count):
            for row, label in enumerate(('before', 'after')):
                image = Image.open(args.output/(label+'-renders')/f'view-{column}-mip-{size}.png')
                tile(sheet, image, column*400, row*430, 400, 430, f'Photo {column}; {label}; native {size}px')
        sheet.save(args.output/f'all-photo-mip-{size}.jpg', quality=96)
    report['contactSheetSHA256'] = {p.name: sha(p) for p in sorted(args.output.glob('*.jpg'))}
    write_json(args.output/'report.json', report)
    print(json.dumps({'output': str(args.output), 'allClayAndMaskImagesIdentical': report['allClayAndMaskImagesIdentical'],
                      'targets': [{k: t[k] for k in ('originalFaceID', 'selection', 'diagnosticRasterVisibleSamples')} for t in targets]}), flush=True)


if __name__ == '__main__':
    main()
