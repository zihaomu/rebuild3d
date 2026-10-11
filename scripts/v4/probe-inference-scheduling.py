#!/usr/bin/env python3
"""In-memory scheduling experiments against a frozen production runtime.

Never edits the packaged runtime. Output arrays must match the specified
candidate reference exactly. Timings are diagnostic, not application gates.
"""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import shutil
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--size', choices=[392, 518], type=int, default=518)
    parser.add_argument('--mode', choices=['control', 'profile-load', 'async-blocks',
                                         'nonblocking-heads', 'batched-load'], required=True)
    args = parser.parse_args()
    runtime, out = args.runtime.resolve(), args.output.resolve()
    assert Path(sys.executable).resolve() == (runtime / 'python/bin/python3.12').resolve()
    manifest_sha = sha(runtime / 'runtime.json')
    assert manifest_sha == '1796bc7e6bff7fc2c59397fcef304268b2cf71c11915b54169425846cfb95069'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    identities = {str(p): sha(p) for p in args.inputs.resolve().rglob('*') if p.is_file()}
    expected_hash = sha(args.reference)
    records = json.loads((args.inputs / 'dataset.json').read_text())['records']
    reference_config = json.loads(args.reference.with_name('config.json').read_text())
    assert 3 <= len(records) <= 12
    assert reference_config['arguments']['size'] == args.size
    assert reference_config['photoIDs'] == [r['id'] for r in records]
    assert reference_config['sourceSHA256'] == [r['sourceSHA256'] for r in records]
    os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='0', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime / 'worker'))
    from rebuild3d_worker import infer, mlx_backend, mlx_aggregator, mlx_block
    import mlx.core as mx
    import numpy as np
    import torch
    for module in (infer, mlx_backend, mlx_aggregator, mlx_block):
        p = Path(module.__file__).resolve()
        assert p.is_relative_to(runtime / 'worker')
        entry = next(e for e in json.loads((runtime/'runtime.json').read_text())['artifacts']
                     if e['path'] == str(p.relative_to(runtime)))
        assert sha(p) == entry['sha256']
    report = {'status': 'running', 'scope': __doc__, 'mode': args.mode,
              'runtimeManifestSHA256': manifest_sha, 'scriptSHA256': sha(__file__),
              'reference': str(args.reference.resolve()), 'referenceSHA256': expected_hash,
              'inputSHA256': identities, 'size': args.size, 'count': len(records),
              'referenceConfigSHA256': sha(args.reference.with_name('config.json'))}
    original_init = mlx_backend.MLXBackend.__init__

    if args.mode == 'profile-load':
        def profiled_init(self, *a, **kw):
            profiler = cProfile.Profile()
            try:
                return profiler.runcall(original_init, self, *a, **kw)
            finally:
                profiler.dump_stats(str(out / 'load.prof'))
                with (out / 'load-profile.txt').open('w') as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(65)
        mlx_backend.MLXBackend.__init__ = profiled_init
    elif args.mode == 'nonblocking-heads':
        def asynchronous_init(self, *a, **kw):
            original_to = torch.Tensor.to
            sources = []
            def transfer(tensor, *ta, **tk):
                if ta and isinstance(ta[0], str) and ta[0] == 'mps' and tensor.device.type == 'cpu':
                    sources.append(tensor)
                    tk = {**tk, 'non_blocking': True}
                return original_to(tensor, *ta, **tk)
            torch.Tensor.to = transfer
            try:
                original_init(self, *a, **kw)
                torch.mps.synchronize()
                report['asynchronousHeadCopies'] = len(sources)
            finally:
                torch.Tensor.to = original_to
                torch.mps.synchronize()
                sources.clear()
        mlx_backend.MLXBackend.__init__ = asynchronous_init
    elif args.mode == 'batched-load':
        # Suppress only the synchronous evaluation inside Block construction.
        original_block_init = mlx_block.Block.__init__
        deferred = []
        def batch_block(self, *a, **kw):
            original_eval = mx.eval
            def defer(*arrays):
                deferred.extend(arrays)
            mx.eval = defer
            try:
                original_block_init(self, *a, **kw)
            finally:
                mx.eval = original_eval
        mlx_block.Block.__init__ = batch_block
        def batched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            mx.eval(deferred)
            report['deferredBlockCollections'] = len(deferred)
            deferred.clear()
        mlx_backend.MLXBackend.__init__ = batched_init
    elif args.mode == 'async-blocks':
        original_call = mlx_aggregator.Aggregator.__call__
        def async_call(self, *a, **kw):
            original_eval = mx.eval
            mx.eval = mx.async_eval
            try:
                result = original_call(self, *a, **kw)
                mx.synchronize()
                return result
            finally:
                mx.eval = original_eval
        mlx_aggregator.Aggregator.__call__ = async_call

    started = time.monotonic()
    try:
        sys.argv = ['scheduling-probe', str(args.inputs.resolve()), str(out/'inference'),
                    '--repo', str(runtime/'vggt'), '--weights', str(runtime/'models/vggt-1b.safetensors'),
                    '--count', str(len(records)), '--size', str(args.size),
                    '--max-memory-gib', '10.5', '--max-seconds', '300']
        infer.main()
        checks = {}
        with np.load(args.reference) as reference, np.load(out/'inference/predictions.npz') as result:
            assert reference.files == result.files
            for key in reference.files:
                a, b = reference[key], result[key]
                checks[key] = {'arrayEqual': a.dtype == b.dtype and bool(np.array_equal(a, b)), 'shape': list(b.shape),
                               'maxAbsolute': float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))}
        report['arrayChecks'] = checks
        report['allArraysExactlyEqual'] = all(v['arrayEqual'] for v in checks.values())
        report['inferenceReport'] = json.loads((out/'inference/report.json').read_text())
        report['pixelTransformsExactlyEqual'] = (
            json.loads((out/'inference/pixel-transforms.json').read_text()) ==
            json.loads(args.reference.with_name('pixel-transforms.json').read_text()))
        assert report['pixelTransformsExactlyEqual']
        assert all(sha(p) == h for p, h in identities.items()) and sha(args.reference) == expected_hash
        report['originalInputsAndReferenceUnchanged'] = True
        report['status'] = 'diagnostic-complete' if report['allArraysExactlyEqual'] else 'rejected-output-difference'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['probeElapsedSeconds'] = time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'status': report['status'], 'mode': args.mode,
                          'allArraysExactlyEqual': report.get('allArraysExactlyEqual'),
                          'timings': report.get('inferenceReport', {}).get('timingsSeconds')}), flush=True)


if __name__ == '__main__':
    main()
