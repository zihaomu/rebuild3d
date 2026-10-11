#!/usr/bin/env python3
"""Exercise real MPS allocation failure and the coordinator's one 392 retry.

Requires the disclosed test-only environment-forwarding probe. Limits apply to
this probe's worker processes, not the user's apps or the whole machine.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np

MATRIX = Path(__file__).with_name('run-input-matrix.py')
SPEC = importlib.util.spec_from_file_location('input_matrix', MATRIX)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
H = M.H
KEYS = ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO')


def load(path):
    return json.loads(path.read_text())


def validate_392(job, photos):
    ids = [p['id'] for p in photos]
    task = load(job / 'request.json')
    assert task['parameters'] == {'memoryGiB': 10.5, 'modelSize': 392,
                                  'volumeResolution': 256, 'textureSize': 4096}
    assert [p['id'] for p in task['photos']] == ids and task['photos'] == task['allPhotos']
    config = load(job / 'inference/config.json')
    assert config['photoIDs'] == ids
    assert config['sourceSHA256'] == [p['metadata']['contentSHA256'] for p in photos]
    inputs = load(job / 'inputs/dataset.json')['records']
    cameras = load(job / 'bundle/cameras.json')['cameras']
    transforms = load(job / 'inference/pixel-transforms.json')
    assert [p['id'] for p in inputs] == [p['id'] for p in cameras] == ids
    assert [p['photoID'] for p in transforms] == ids
    errors = []
    with np.load(job / 'inference/predictions.npz') as arrays:
        for key in ('depth', 'confidence', 'extrinsics', 'intrinsics', 'rgb', 'masks'):
            assert arrays[key].shape[0] == len(photos) and np.isfinite(arrays[key]).all()
        assert arrays['depth'].shape == (len(photos), 392, 392)
        for i, (r, t, c, photo) in enumerate(zip(inputs, transforms, cameras, photos)):
            assert r['encodedOrientation'] == photo['orientation']
            w, h = r['encodedWidth'], r['encodedHeight']
            uw, uh = (h, w) if r['encodedOrientation'] >= 5 else (w, h)
            assert [uw, uh] == [r['uprightWidth'], r['uprightHeight']] == c['uprightSize']
            ww, wh = r['workingWidth'], r['workingHeight']
            rotation = M.orientation_matrix(w, h, r['encodedOrientation'])
            working = M.affine_resize(uw, uh, ww, wh) @ rotation
            nw, nh = round(ww * 392 / max(ww, wh)), round(wh * 392 / max(ww, wh))
            model = M.affine_resize(ww, wh, nw, nh)
            model[:2, 2] += [(392 - nw) // 2, (392 - nh) // 2]
            np.testing.assert_allclose(r['encodedToWorkingPixels'], working.reshape(-1), atol=1e-9, rtol=0)
            np.testing.assert_allclose(t['workingToModelPixels'], model, atol=1e-9, rtol=0)
            np.testing.assert_allclose(t['encodedToModelPixels'], model @ working, atol=1e-9, rtol=0)
            k = arrays['intrinsics'][i]
            work_k = np.linalg.inv(model) @ k
            full_k = np.linalg.inv(M.affine_resize(uw, uh, ww, wh)) @ work_k
            np.testing.assert_allclose(c['workingIntrinsics'], work_k, atol=1e-5, rtol=1e-7)
            np.testing.assert_allclose(c['fullIntrinsics'], full_k, atol=1e-5, rtol=1e-7)
            np.testing.assert_allclose(c['worldToCamera'], arrays['extrinsics'][i], atol=1e-7, rtol=1e-7)
            rays = np.array([[-.4, -.2, 1], [0, 0, 1], [.3, .25, 1]], float).T
            error = float(np.max(np.abs(model @ working @ np.linalg.inv(rotation) @ full_k @ rays - k @ rays)))
            assert error < 1e-5
            errors.append(error)
    with np.load(job / 'bundle/geometry.npz') as geometry:
        assert len(geometry['faces']) and np.isfinite(geometry['vertices']).all()
    events = [json.loads(line) for line in (job / 'events.jsonl').read_text().splitlines()]
    assert not any(e['event'] == 'stage-reused' for e in events)
    return {'photoCount': len(photos), 'maxProjectionChainErrorPixels': max(errors),
            'freshStageReuseCount': 0, 'parameters': task['parameters']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--probe-build', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-store', type=Path, required=True)
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--high-ratio', type=float, default=.215)
    parser.add_argument('--low-ratio', type=float, default=.20)
    parser.add_argument('--stop-if-untriggered', action='store_true',
                        help='Cancel after a successful 518 inference; retain this as a non-triggered test')
    args = parser.parse_args()
    assert 0 < args.low_ratio <= args.high_ratio <= 1
    output, source, runtime = args.output.resolve(), args.source_project.resolve(), args.runtime.resolve()
    build = args.probe_build.resolve()
    build_record = load(build / 'build.json')
    probe = build / 'probe-resource-retry'
    assert build_record['exitCode'] == 0 and H.sha(probe) == build_record['binarySHA256']
    output.mkdir(parents=True, exist_ok=False)
    for script in (Path(__file__), MATRIX, M.HELPER, Path(__file__).with_name('audit-application-export.py')):
        shutil.copy2(script, output / script.name)
    shutil.copy2(build / 'build.json', output / 'probe-build.json')
    before_source = H.snapshot(source)
    project = output / 'B-resource-retry.rebuild3d'
    subprocess.run(['/bin/cp', '-cR', str(source), str(project)], check=True)
    before = H.snapshot(project)
    assert before == before_source
    H.save(output / 'protected-before.json', before)
    initial = set((project / 'generation').glob('*/request.json'))
    limits = dict(zip(KEYS, (str(args.high_ratio), str(args.low_ratio))))
    report = {'status': 'running', 'startedAtUTC': H.now(), 'runtimeManifestSHA256': H.sha(runtime / 'runtime.json'),
              'project': str(project), 'sourceProject': str(source), 'scriptSHA256': H.sha(__file__),
              'probeBuildSHA256': H.sha(build / 'build.json'), 'probeSHA256': H.sha(probe),
              'workerEnvironmentLimits': limits,
              'allocatorReference': 'https://docs.pytorch.org/docs/2.14/mps_environment_variables.html',
              'scope': 'Real allocator failure under a controlled per-process limit; test probe modifies only environment forwarding',
              'limits': ['Not a system-wide physical low-memory test', 'Not an MLX allocator-specific failure test',
                         'Not actual UI, app restart, offline or performance acceptance'],
              'expectedFlow': '518 real resource failure, one 392 attempt, full result publication and export'}
    query = [str(runtime / 'python/bin/python3.12'), '-I', '-B', '-c',
             'import torch,json; print(json.dumps({"torch":torch.__version__,"mpsAvailable":torch.backends.mps.is_available(),"recommendedMaxMemoryBytes":torch.mps.recommended_max_memory()}))']
    checked = subprocess.run(query, capture_output=True, text=True, timeout=30,
                             env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert checked.returncode == 0, checked.stderr
    report['deviceQuery'] = {'command': query, 'exitCode': checked.returncode, 'result': json.loads(checked.stdout)}
    assert report['deviceQuery']['result']['mpsAvailable']
    report['configuredMPSHighLimitBytes'] = int(report['deviceQuery']['result']['recommendedMaxMemoryBytes'] * args.high_ratio)
    H.save(output / 'report.json', report)
    observations, watch_errors, early_stop = [], [], []
    stop = threading.Event()

    def watch_retry():
        while not stop.wait(.1):
            try:
                jobs = [p.parent for p in set((project / 'generation').glob('*/request.json')) - initial]
                sizes = {load(p / 'request.json')['parameters']['modelSize']: p for p in jobs}
                if args.stop_if_untriggered and set(sizes) == {518} and (sizes[518] / 'inference/report.json').exists():
                    first = load(sizes[518] / 'inference/report.json')
                    if first['status'] == 'inference-complete':
                        record = {'atUTC': H.now(), 'reason': '518 inference completed; allocator retry did not trigger',
                                  'inferenceReportSHA256': H.sha(sizes[518] / 'inference/report.json')}
                        early_stop.append(record)
                        H.save(output / 'untriggered-cancellation.json', record)
                        (output / 'generated/request-cancel').write_text(H.now() + '\n')
                        return
                if set(sizes) == {518, 392} and (sizes[518] / 'inference/report.json').exists():
                    failed = load(sizes[518] / 'inference/report.json')
                    if failed['status'] == 'resource-limit':
                        current = H.snapshot(project)
                        H.save(output / 'protected-during-retry.json', current)
                        observations.append({'atUTC': H.now(), 'oldPublishedFilesUnchanged': current == before,
                                             'firstInferenceReportSHA256': H.sha(sizes[518] / 'inference/report.json')})
                        return
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            except Exception as error:
                watch_errors.append(repr(error))
                return

    thread = threading.Thread(target=watch_retry, daemon=True)
    old_env = {k: os.environ.get(k) for k in KEYS}
    os.environ.update(limits)
    thread.start()
    exit_code = 1
    try:
        result = H.run_phase(probe, runtime, args.runtime_store.resolve(), project, output / 'generated', None)
        stop.set()
        thread.join()
        jobs = [p.parent for p in set((project / 'generation').glob('*/request.json')) - initial]
        sizes = {load(p / 'request.json')['parameters']['modelSize']: p for p in jobs}
        report.update(coordinatorExitCode=result['exitCode'], jobCount=len(jobs),
                      jobs={str(k): str(v) for k, v in sizes.items()}, retryObservations=observations,
                      observationErrors=watch_errors, earlyStop=early_stop)
        if len(jobs) == 1 and set(sizes) == {518} and (result['exitCode'] == 0 or (result['exitCode'] == 130 and early_stop)):
            assert load(sizes[518] / 'inference/report.json')['status'] == 'inference-complete'
            if early_stop:
                assert H.snapshot(project) == before
                report['oldResultProtected'] = True
            report['status'] = 'limit-did-not-trigger-retry'
            exit_code = 3
        else:
            assert len(jobs) == 2 and set(sizes) == {518, 392}
            first = load(sizes[518] / 'inference/report.json')
            assert first['status'] == 'resource-limit' and first['resourceFailureReason'] == 'pytorch-allocation-failed'
            assert 'MPS backend out of memory' in first['error']
            report['realAllocatorFailure'] = {k: first[k] for k in ['status', 'resourceFailureReason', 'error']}
            assert observations and observations[0]['oldPublishedFilesUnchanged'] and not watch_errors
            if result['exitCode'] != 0:
                second = load(sizes[392] / 'inference/report.json')
                assert second['status'] == 'resource-limit'
                assert H.snapshot(project) == before
                report.update(status='both-attempts-resource-limited', oldResultProtected=True,
                              secondFailure={k: second[k] for k in ['status', 'error']})
                exit_code = 3
            else:
                assert load(sizes[392] / 'state.json')['status'] == 'completed'
                assert load(sizes[392] / 'inference/report.json')['status'] == 'inference-complete'
                old_manifest, new_manifest = load(source / 'project.json'), load(project / 'project.json')
                assert old_manifest['photos'] == new_manifest['photos']
                assert old_manifest['model']['path'] != new_manifest['model']['path']
                report['coordinates'] = validate_392(sizes[392], new_manifest['photos'])
                after = H.snapshot(project)
                assert H.unchanged(before, after, except_manifest=True)
                H.save(output / 'protected-after-publication.json', after)
                command = [str(runtime / 'python/bin/python3.12'), '-B',
                           str(Path(__file__).with_name('audit-application-export.py').resolve()),
                           str(project), str(output / 'generated/recovered.usdz'), str(output / 'export-audit')]
                with (output / 'export-audit.log').open('w') as log:
                    audit = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'})
                H.save(output / 'export-process.json', {'command': command, 'exitCode': audit.returncode})
                assert audit.returncode == 0
                report.update(status='controlled-allocator-core-retry-passed', job=str(sizes[392]),
                              photoCount=len(new_manifest['photos']), oldResultProtectedDuringRetry=True,
                              oldArtifactsRetainedAfterPublication=True, exportAuditSHA256=H.sha(output / 'export-audit/report.json'))
                exit_code = 0
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        stop.set()
        thread.join()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report['originalSourceUnchanged'] = H.snapshot(source) == before_source
        report['finishedAtUTC'] = H.now()
        H.save(output / 'report.json', report)
        assert report['originalSourceUnchanged']
    print(json.dumps({'status': report['status'], 'output': str(output)}), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
