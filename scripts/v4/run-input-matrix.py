#!/usr/bin/env python3
"""Fresh production import/generation for 3–11 registered D photographs and order variants.

The existing D12 acceptance is recorded separately. No UI or performance claim.
Source files and projects are never rewritten. Every output root must be new.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

HELPER = Path(__file__).with_name('run-generation-recovery.py')
SPEC = importlib.util.spec_from_file_location('recovery_harness', HELPER)
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)


def affine_resize(width, height, new_width, new_height):
    sx, sy = new_width / width, new_height / height
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])


def orientation_matrix(width, height, orientation):
    x, y = width - 1, height - 1
    return np.array({
        1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        2: [[-1, 0, x], [0, 1, 0], [0, 0, 1]],
        3: [[-1, 0, x], [0, -1, y], [0, 0, 1]],
        4: [[1, 0, 0], [0, -1, y], [0, 0, 1]],
        5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        6: [[0, -1, y], [1, 0, 0], [0, 0, 1]],
        7: [[0, -1, y], [-1, 0, x], [0, 0, 1]],
        8: [[0, 1, 0], [-1, 0, x], [0, 0, 1]],
    }[orientation], dtype=float)


def validate_job(project, expected_sources):
    manifest = json.loads((project / 'project.json').read_text())
    photos = manifest['photos']
    count = len(expected_sources)
    ids = [p['id'] for p in photos]
    names = [Path(p['path']).name for p in expected_sources]
    digests = [p['sha256'] for p in expected_sources]
    assert [p['originalName'] for p in photos] == names
    assert [p['metadata']['contentSHA256'] for p in photos] == digests
    assert manifest['model']['photoIDs'] == ids and manifest['model']['approximation']
    jobs = list((project / 'generation').glob('*/request.json'))
    assert len(jobs) == 1, 'Expected a fresh, single 518 attempt; inspect unexpected retry'
    job = jobs[0].parent
    task = json.loads(jobs[0].read_text())
    assert task['parameters'] == {'memoryGiB': 10.5, 'modelSize': 518,
                                   'textureSize': 4096, 'volumeResolution': 256}
    assert [p['id'] for p in task['photos']] == ids
    assert task['photos'] == task['allPhotos']
    events = [json.loads(line) for line in (job / 'events.jsonl').read_text().splitlines()]
    assert not any(e['event'] == 'stage-reused' for e in events)
    identity = json.loads((job / 'runtime-identity.json').read_text())
    assert identity['backendID'] == 'mlx-aggregator-pytorch-heads-v1'
    config = json.loads((job / 'inference/config.json').read_text())
    assert config['photoIDs'] == ids and config['sourceSHA256'] == digests
    dataset = json.loads((job / 'inputs/dataset.json').read_text())['records']
    transforms = json.loads((job / 'inference/pixel-transforms.json').read_text())
    cameras = json.loads((job / 'bundle/cameras.json').read_text())['cameras']
    assert [r['id'] for r in dataset] == [c['id'] for c in cameras] == ids
    assert [t['photoID'] for t in transforms] == ids
    assert [r['name'] for r in dataset] == [c['name'] for c in cameras] == names
    geometry = np.load(job / 'bundle/geometry.npz')
    assert len(geometry['faces']) > 0 and np.isfinite(geometry['vertices']).all()
    geometry.close()
    errors = []
    with np.load(job / 'inference/predictions.npz') as arrays:
        for key in ('depth', 'confidence', 'extrinsics', 'intrinsics', 'rgb', 'masks'):
            assert arrays[key].shape[0] == count and np.isfinite(arrays[key]).all(), key
        assert arrays['depth'].shape == (count, 518, 518)
        for i, (r, t, c, photo) in enumerate(zip(dataset, transforms, cameras, photos)):
            assert r['encodedOrientation'] == photo['orientation']
            w, h = r['encodedWidth'], r['encodedHeight']
            uw, uh = (h, w) if r['encodedOrientation'] >= 5 else (w, h)
            assert [uw, uh] == [r['uprightWidth'], r['uprightHeight']] == c['uprightSize']
            ww, wh = r['workingWidth'], r['workingHeight']
            assert c['workingSize'] == [ww, wh]
            rotation = orientation_matrix(w, h, r['encodedOrientation'])
            working = affine_resize(uw, uh, ww, wh) @ rotation
            nw, nh = round(ww * 518 / max(ww, wh)), round(wh * 518 / max(ww, wh))
            model = affine_resize(ww, wh, nw, nh)
            model[:2, 2] += [(518 - nw) // 2, (518 - nh) // 2]
            np.testing.assert_allclose(r['encodedToUprightPixels'], rotation.reshape(-1), atol=1e-9, rtol=0)
            np.testing.assert_allclose(r['encodedToWorkingPixels'], working.reshape(-1), atol=1e-9, rtol=0)
            np.testing.assert_allclose(t['workingToModelPixels'], model, atol=1e-9, rtol=0)
            np.testing.assert_allclose(t['encodedToModelPixels'], model @ working, atol=1e-9, rtol=0)
            k = arrays['intrinsics'][i]
            work_k = np.linalg.inv(model) @ k
            full_k = np.linalg.inv(affine_resize(uw, uh, ww, wh)) @ work_k
            np.testing.assert_allclose(c['workingIntrinsics'], work_k, atol=1e-5, rtol=1e-7)
            np.testing.assert_allclose(c['fullIntrinsics'], full_k, atol=1e-5, rtol=1e-7)
            np.testing.assert_allclose(c['worldToCamera'], arrays['extrinsics'][i], atol=1e-7, rtol=1e-7)
            # Camera rays projected through encoded/working/model coordinates agree.
            rays = np.array([[-.4, -.2, 1], [0, 0, 1], [.3, .25, 1]], float).T
            expected = k @ rays
            actual = model @ working @ np.linalg.inv(rotation) @ full_k @ rays
            error = float(np.max(np.abs(actual - expected)))
            assert error < 1e-5
            errors.append(error)
    return {'job': str(job), 'photoCount': count, 'photoIDs': ids, 'orientations': [p['orientation'] for p in photos],
            'pixelTransformsAndCameraFramesVerified': True, 'maxProjectionChainErrorPixels': max(errors),
            'freshStageReuseCount': 0, 'parameters': task['parameters'],
            'runtimeIdentitySHA256': H.sha(job / 'runtime-identity.json'),
            'modelSHA256': H.sha(project / manifest['model']['path'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--import-probe', required=True, type=Path)
    parser.add_argument('--generation-probe', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--runtime-store', required=True, type=Path)
    parser.add_argument('--source-project', type=Path, default=Path('build/v3-oneclick/delivery/D-skull-v3-auto.rebuild3d'))
    parser.add_argument('--counts', type=int, nargs='+', default=list(range(3, 12)))
    parser.add_argument('--reverse-seven', action='store_true')
    args = parser.parse_args()
    assert args.counts and len(set(args.counts)) == len(args.counts) and all(3 <= n <= 12 for n in args.counts)
    output, source, runtime = args.output.resolve(), args.source_project.resolve(), args.runtime.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for script in (Path(__file__), HELPER, Path(__file__).with_name('audit-application-export.py')):
        shutil.copy2(script, output / script.name)
    source_manifest = json.loads((source / 'project.json').read_text())
    registered = json.loads(Path('build/v3-oneclick/heldout/skull-inputs.json').read_text())
    photos = source_manifest['photos']
    assert [(p['originalName'], p['metadata']['contentSHA256']) for p in photos] == [
        (p['localName'], p['sha256']) for p in registered]
    source_before = {p['imagePath']: H.sha(source / p['imagePath']) for p in photos}
    assert all(source_before[p['imagePath']] == p['metadata']['contentSHA256'] for p in photos)
    manifest_sha = H.sha(source / 'project.json')
    report = {'status': 'running', 'startedAtUTC': H.now(), 'scope': 'Production import/core generation; fresh D subsets',
              'runtimeManifestSHA256': H.sha(runtime / 'runtime.json'), 'scriptSHA256': H.sha(__file__),
              'helperSHA256': H.sha(HELPER), 'importProbeSHA256': H.sha(args.import_probe),
              'generationProbeSHA256': H.sha(args.generation_probe), 'records': [],
              'sourceManifestSHA256': manifest_sha,
              'selectionRule': 'Floor(i * 12 / n), i=0..n-1; same rule frozen before any subset run',
              'limits': ['No UI/offline/performance acceptance', 'Not ground-truth geometry validation',
                         'All original source orientations are recorded; generated EXIF variants are separate',
                         'Held-out quality is tracked separately; these are registered D regression inputs', 'Native visual review separate from artifact/coordinate checks']}
    H.save(output / 'report.json', report)
    try:
        cases = [(f'D{n:02d}', [i * 12 // n for i in range(n)]) for n in args.counts]
        if args.reverse_seven:
            cases.append(('D07-reversed', list(reversed([i * 12 // 7 for i in range(7)]))))
        H.save(output / 'cases.json', [{'name': name, 'sourceIndices': indices} for name, indices in cases])
        for name, indices in cases:
            base = output / (name + ' 中文 空格')
            base.mkdir()
            inputs = base / '输入照片'
            inputs.mkdir()
            selected = []
            for index in indices:
                photo = photos[index]
                target = inputs / (f'视角 {index + 1:02d} ({photo["originalName"]})' + '.jpg')
                subprocess.run(['/bin/cp', '-c', str(source / photo['imagePath']), str(target)], check=True)
                assert H.sha(target) == photo['metadata']['contentSHA256']
                selected.append({'path': str(target), 'sha256': H.sha(target), 'sourcePhotoID': photo['id'],
                                 'originalName': photo['originalName']})
            listing = base / 'inputs.json'
            H.save(listing, {'sources': selected})
            project = base / '新项目.rebuild3d'
            record = {'name': name, 'sourceIndices': indices, 'sources': selected, 'project': str(project), 'status': 'running'}
            report['records'].append(record)
            H.save(output / 'report.json', report)
            command = [str(args.import_probe.resolve()), str(listing), str(project)]
            with (base / 'import.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            H.save(base / 'import-process.json', {'command': command, 'exitCode': result.returncode})
            assert result.returncode == 0, f'Import failed: {base}'
            before_photos = json.loads((project / 'project.json').read_text())['photos']
            phase = base / 'generated'
            result = H.run_phase(args.generation_probe.resolve(), runtime, args.runtime_store.resolve(), project, phase, None)
            assert result['exitCode'] == 0, f'Generation failed: {base}'
            assert json.loads((project / 'project.json').read_text())['photos'] == before_photos
            record.update(validate_job(project, selected))
            command = [str(runtime / 'python/bin/python3.12'), '-B', str(Path(__file__).with_name('audit-application-export.py').resolve()),
                       str(project), str(phase / 'recovered.usdz'), str(base / 'export-audit')]
            with (base / 'export-audit.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'})
            H.save(base / 'export-process.json', {'command': command, 'exitCode': result.returncode})
            assert result.returncode == 0, f'Export audit failed: {base}'
            record.update(status='core-input-and-export-passed', exportAuditSHA256=H.sha(base / 'export-audit/report.json'),
                          elapsedSecondsDiagnosticOnly=json.loads((phase / 'process-result.json').read_text())['elapsedSeconds'])
            H.save(output / 'report.json', report)
            print(json.dumps({'name': name, 'status': record['status'], 'photoCount': record['photoCount'],
                              'projectionError': record['maxProjectionChainErrorPixels']}, ensure_ascii=False), flush=True)
        report['status'] = 'core-input-matrix-passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['originalSourcesUnchanged'] = (H.sha(source / 'project.json') == manifest_sha and
            all(H.sha(source / path) == digest for path, digest in source_before.items()))
        report['finishedAtUTC'] = H.now()
        H.save(output / 'report.json', report)
        assert report['originalSourcesUnchanged']


if __name__ == '__main__':
    main()
