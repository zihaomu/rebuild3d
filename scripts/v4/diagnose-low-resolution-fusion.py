#!/usr/bin/env python3
"""Compare preserved 392 inference through unchanged fusion and observe filtering.

Use candidate Python and its worker PYTHONPATH. This is a diagnostic replay,
not new inference, a performance sample, or native application acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import trimesh

from rebuild3d_worker import fuse
from rebuild3d_worker.common import camera_project, raster


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def load_arrays(path):
    with np.load(path / 'predictions.npz') as data:
        return {key: data[key] for key in data.files}


def replay(inference, output, resolution):
    original = fuse.retain_component
    captured = {}

    def observe(*args, **kwargs):
        frame = inspect.currentframe().f_back
        assert frame.f_code is fuse.main.__code__
        local = frame.f_locals
        if not captured:
            captured.update(vertices=local['vertices'].copy(), faces=local['mesh'].faces.copy(),
                            retained=np.zeros(len(local['mesh'].faces), dtype=bool))
        result = original(*args, **kwargs)
        if result[0]:
            captured['retained'][local['component']] = True
        return result

    fuse.retain_component = observe
    try:
        sys.argv = ['unchanged-fusion-replay', str(inference), str(output),
                    '--resolution', str(resolution)]
        fuse.main()
    finally:
        fuse.retain_component = original
    np.savez_compressed(output / 'before-component-filter.npz', **captured)
    return captured


def draw_mesh(vertices, faces, ext, intr, size, labels=None):
    _, ids, _ = raster(camera_project(vertices, ext, intr), faces, size, size)
    active = ids >= 0
    image = np.zeros((size, size, 3), dtype='uint8')
    if labels is not None:
        kept = active.copy()
        kept[active] = labels[ids[active]]
        image[kept] = [70, 210, 100]
        image[active & ~kept] = [245, 80, 70]
    else:
        normals = trimesh.Trimesh(vertices=vertices, faces=faces,
                                  process=False).face_normals @ ext[:, :3].T
        light = np.array([-.25, -.35, -1.])
        light /= np.linalg.norm(light)
        shade = np.clip((.3 + .65 * np.maximum(normals @ light, 0)) * 240, 0, 255)
        image[active] = shade[ids[active], None].astype('uint8')
    return image, active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_job', type=Path)
    parser.add_argument('prior_candidate_inference', type=Path)
    parser.add_argument('floating_inference', type=Path)
    parser.add_argument('reference_518_job', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    report = {'status': 'running', 'startedAtUTC': datetime.now(timezone.utc).isoformat(),
              'scriptSHA256': sha(__file__), 'fusionSHA256': sha(fuse.__file__),
              'scope': 'Observed unchanged fusion replay; preserved inference reused after equality checks',
              'nativeVisualAcceptance': False}
    save(out / 'report.json', report)
    try:
        job = args.candidate_job.resolve()
        inf = job / 'inference'
        prior = args.prior_candidate_inference.resolve()
        floating = args.floating_inference.resolve()
        ref = args.reference_518_job.resolve()
        arrays = [load_arrays(p) for p in (inf, prior, floating, ref / 'inference')]
        configs = [json.loads((p / 'config.json').read_text())
                   for p in (inf, prior, floating, ref / 'inference')]
        assert all(c['sourceSHA256'] == configs[0]['sourceSHA256'] for c in configs)
        assert all(c['modelImageSize'] == 392 for c in configs[:3])
        assert configs[3]['modelImageSize'] == 518
        assert configs[2]['backendID'] == 'pytorch-mps-v3'
        exact = {key: bool(np.array_equal(arrays[0][key], arrays[1][key])) for key in arrays[0]}
        assert all(exact.values())
        assert all(np.array_equal(arrays[0][key], arrays[2][key]) for key in ('rgb', 'masks'))
        report.update(currentEqualsPriorCandidate=exact, floatingInputsExactlyEqual=True,
                      orderedSourcesExactlyEqual=True,
                      inputs={str(p): {'configSHA256': sha(p / 'config.json'),
                                      'predictionsSHA256': sha(p / 'predictions.npz')}
                              for p in (inf, prior, floating, ref / 'inference')})
        save(out / 'report.json', report)
        resolution = json.loads((job / 'fusion/report.json').read_text())['resolution']
        captured = replay(inf, out / 'quantized-fusion', resolution)
        with np.load(job / 'fusion/face-provenance.npz') as old, \
                np.load(out / 'quantized-fusion/face-provenance.npz') as new:
            same = {k: bool(np.array_equal(old[k], new[k])) for k in old.files}
            assert set(old.files) == set(new.files) and all(same.values())
        report['observedReplayExactlyMatchesCurrentFusion'] = same
        save(out / 'report.json', report)
        replay(floating, out / 'floating-fusion', resolution)
        mesh_data = []
        for path in (out / 'quantized-fusion', out / 'floating-fusion', ref / 'fusion'):
            with np.load(path / 'face-provenance.npz') as data:
                mesh_data.append((data['verticesBeforeDisplay'], data['faces']))
        rows = []
        sheet = Image.new('RGB', (6 * 280, 40 + 300 * 7), '#161616')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), 'Input | mask | before filter: kept green / removed red | W8 392 | float 392 | W8 518', fill='white')
        for i in range(7):
            q, _, f, h = arrays
            before, before_mask = draw_mesh(captured['vertices'], captured['faces'],
                                            q['extrinsics'][i], q['intrinsics'][i], 392,
                                            labels=captured['retained'])
            final_images, final_masks = [], []
            for (vertices, faces), data, size in zip(mesh_data, (q, f, h), (392, 392, 518)):
                im, mask = draw_mesh(vertices, faces, data['extrinsics'][i], data['intrinsics'][i], size)
                final_images.append(im)
                final_masks.append(mask)
            foreground = q['rgb'][i].copy()
            foreground[~q['masks'][i]] = 0
            images = [q['rgb'][i], foreground, before, *final_images]
            for col, im in enumerate(images):
                image = Image.fromarray(im)
                image.save(out / f'view-{i}-col-{col}.png')
                tile = ImageOps.contain(image, (276, 276))
                sheet.paste(tile, (col * 280 + 2, 60 + i * 300))
            draw.text((5, 42 + i * 300), f'Photo {i + 1}', fill='white')
            rows.append({'photoIndex': i, 'beforeFilterPixels': int(before_mask.sum()),
                         'retainedPixels': int(final_masks[0].sum()),
                         'floating392Pixels': int(final_masks[1].sum()),
                         'candidateMaskIoU': float((final_masks[0] & q['masks'][i]).sum() /
                                                 (final_masks[0] | q['masks'][i]).sum()),
                         'floatingMaskIoU': float((final_masks[1] & f['masks'][i]).sum() /
                                                (final_masks[1] | f['masks'][i]).sum())})
        sheet.save(out / 'filter-comparison.jpg', quality=95)
        report.update(status='diagnostic-completed-visual-review-pending', views=rows,
                      comparisonSheetSHA256=sha(out / 'filter-comparison.jpg'))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finishedAtUTC'] = datetime.now(timezone.utc).isoformat()
        save(out / 'report.json', report)


if __name__ == '__main__':
    main()
