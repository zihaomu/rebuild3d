#!/usr/bin/env python3
"""Supplement legacy v3 timing with synchronized function measurements.

Runs the frozen baseline infer.main with temporary in-memory timing wrappers.
Checks every prediction array against an existing unchanged baseline job.
Additional synchronization changes scheduling: these are diagnostic substage
times, never substitutes for the unmodified real-UI performance samples.
Run with the frozen runtime's own Python, after other heavy jobs have finished.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_job', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('build/v4-quantization/baseline/protocol.json'))
    args = parser.parse_args()
    runtime, job, out = args.runtime.resolve(), args.baseline_job.resolve(), args.output.resolve()
    protocol = json.loads(args.protocol.read_text())
    assert sha(runtime / 'runtime.json') == protocol['runtimeManifestSHA256']
    assert Path(sys.executable).resolve() == (runtime / 'python/bin/python3.12').resolve()
    for artifact in json.loads((runtime / 'runtime.json').read_text())['artifacts']:
        path = runtime / artifact['path']
        assert path.resolve().is_relative_to(runtime)
        assert path.stat().st_size == artifact['byteCount'] and sha(path) == artifact['sha256']
    original_config = json.loads((job / 'inference/config.json').read_text())
    assert original_config['arguments']['size'] == 518 and len(original_config['photoIDs']) == 7
    sources = {str(p): sha(p) for p in (job / 'inputs').rglob('*') if p.is_file()}
    expected = job / 'inference/predictions.npz'
    original_prediction_sha = sha(expected)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='0', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime / 'worker'))
    imported_at = time.monotonic()
    from rebuild3d_worker import infer
    import torch
    import numpy as np
    import_seconds = time.monotonic() - imported_at
    assert Path(infer.__file__).resolve() == runtime / 'worker/rebuild3d_worker/infer.py'
    assert torch.backends.mps.is_available()
    measurements = {}

    def wrap(name, original, gpu=False):
        def measured(*a, **kw):
            if gpu:
                torch.mps.synchronize()
            started = time.monotonic()
            result = original(*a, **kw)
            if gpu:
                torch.mps.synchronize()
            measurements.setdefault(name, []).append(time.monotonic() - started)
            return result
        return measured

    infer.verify_repository = wrap('verifyRepository', infer.verify_repository)
    digest = infer.digest

    def timed_digest(path):
        if Path(path).resolve() == runtime / 'models/vggt-1b.safetensors':
            return wrap('verifyWeightDigest', digest)(path)
        return digest(path)

    infer.digest = timed_digest
    infer.prepare = wrap('prepareInputs', infer.prepare)
    load_model = wrap('loadWeights', infer.load_model, gpu=True)

    def timed_load(*a, **kw):
        model, ignored = load_model(*a, **kw)
        for name in ('aggregator', 'camera_head', 'depth_head'):
            module = getattr(model, name)
            module.forward = wrap(name + 'Forward', module.forward, gpu=True)
        return model, ignored

    infer.load_model = timed_load
    report = {'status': 'running', 'scope': __doc__, 'startedAtUTC': datetime.now(timezone.utc).isoformat(),
              'runtimeManifestSHA256': sha(runtime / 'runtime.json'), 'scriptSHA256': sha(__file__),
              'baselineJob': str(job), 'baselinePredictionsSHA256': original_prediction_sha,
              'baselineInferSourceSHA256': sha(infer.__file__), 'moduleImportSeconds': import_seconds}
    try:
        sys.argv = ['profile-legacy-inference', str(job / 'inputs'), str(out / 'inference'),
                    '--repo', str(runtime / 'vggt'), '--weights', str(runtime / 'models/vggt-1b.safetensors'),
                    '--size', '518', '--count', '7', '--max-memory-gib', '10.5']
        infer.main()
        checks = {}
        with np.load(expected) as old, np.load(out / 'inference/predictions.npz') as new:
            assert old.files == new.files
            for key in old.files:
                equal = np.array_equal(old[key], new[key])
                checks[key] = {'equal': equal, 'shape': list(new[key].shape)}
            assert all(v['equal'] for v in checks.values()), 'Timing instrumentation changed predictions'
        assert sha(expected) == original_prediction_sha
        assert all(sha(p) == digest for p, digest in sources.items())
        assert all(len(v) == 1 for v in measurements.values())
        report.update(status='supplemental-substage-profile-passed', predictionArrayChecks=checks,
                      originalInputsAndPredictionsUnchanged=True,
                      synchronizedNetworkForwardSeconds=sum(measurements[k][0] for k in
                          ('aggregatorForward', 'camera_headForward', 'depth_headForward')),
                      limits=['First execution includes compilation; no steady-state or UI timing claim',
                              'Forward timings exclude CPU readback and include extra synchronization boundaries',
                              'Import time is separate; interpreter startup and compiler-only costs are not independently isolated'])
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report.update(measurementsSeconds=measurements, finishedAtUTC=datetime.now(timezone.utc).isoformat())
        (out / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
