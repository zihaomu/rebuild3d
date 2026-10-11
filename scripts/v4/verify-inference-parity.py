#!/usr/bin/env python3
"""Verify updated worker predictions against existing, unchanged reference jobs.

Uses the specified packaged Python, weights and VGGT sources with either a
frozen worker source overlay or the actual packaged worker and bytecode.
Prepared inputs are reused only for correctness; this does not establish
new-task, UI, recovery or performance acceptance.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

HELPER = Path(__file__).with_name('benchmark.py')
SPEC = importlib.util.spec_from_file_location('benchmark', HELPER)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path, help='JSON list of name and inference directory')
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--worker-root', type=Path, required=True, help='Parent of rebuild3d_worker')
    parser.add_argument('--packaged-worker', action='store_true',
                        help='Use the actual runtime worker and its bytecode instead of copying source files')
    parser.add_argument('--backend-version', type=int, required=True)
    parser.add_argument('--execution-policy', required=True)
    args = parser.parse_args()
    runtime, root = args.runtime.resolve(), args.output.resolve()
    cases = json.loads(args.cases.read_text())
    assert cases and len({c['name'] for c in cases}) == len(cases)
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+', c['name']) for c in cases)
    root.mkdir(parents=True, exist_ok=False)
    if args.packaged_worker:
        if args.worker_root.resolve() != runtime / 'worker':
            raise ValueError('Packaged-worker mode requires this runtime worker directory')
        worker = runtime / 'worker/rebuild3d_worker'
    else:
        worker = root / 'worker/rebuild3d_worker'
        worker.mkdir(parents=True)
        for source in (args.worker_root / 'rebuild3d_worker').glob('*.py'):
            shutil.copy2(source, worker / source.name)
    for path in (Path(__file__), HELPER):
        shutil.copy2(path, root / path.name)
    shutil.copy2(args.cases, root / 'cases.json')
    identities = {p.name: B.sha(p) for p in worker.glob('*.py')}
    report = {'status': 'running', 'scope': __doc__, 'scriptSHA256': B.sha(__file__),
              'helperSHA256': B.sha(HELPER), 'caseManifestSHA256': B.sha(args.cases),
              'runtimeManifestSHA256': B.sha(runtime / 'runtime.json'),
              'workerLoadPolicy': 'packaged-worker' if args.packaged_worker else 'source-overlay',
              'workerSourceSHA256': identities, 'records': [], 'goalComplete': False}
    B.save(root / 'report.json', report)
    try:
        for case in cases:
            reference = Path(case['inference']).resolve()
            old_config = json.loads((reference / 'config.json').read_text())
            arguments = old_config['arguments']
            inputs = Path(arguments['dataset']).resolve()
            folder = root / case['name']
            folder.mkdir()
            output = folder / 'inference'
            original = {str(p): B.sha(p) for p in (reference / 'config.json', reference / 'predictions.npz',
                                                 reference / 'pixel-transforms.json', inputs / 'dataset.json')}
            command = [str(runtime / 'python/bin/python3.12'), '-I', '-B', '-c',
                       'import sys;sys.path.insert(0,sys.argv.pop(1));from rebuild3d_worker.infer import main;main()',
                       str(worker.parent), str(inputs), str(output), '--repo', str(runtime / 'vggt'),
                       '--weights', str(runtime / 'models/vggt-1b.safetensors'),
                       '--size', str(arguments['size']), '--count', str(len(old_config['photoIDs'])),
                       '--max-memory-gib', '10.5', '--max-seconds', '300']
            if arguments.get('foreground'):
                command.append('--foreground')
            env = os.environ.copy()
            for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'REBUILD3D_PARENT_PID',
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO'):
                env.pop(key, None)
            env.update(PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1')
            row = {'name': case['name'], 'command': command, 'before': B.environment(),
                   'referenceAndDatasetSHA256': original}
            report['records'].append(row)
            B.save(root / 'report.json', report)
            print(json.dumps({'started': case['name']}), flush=True)
            with (folder / 'inference.log').open('w') as log:
                row['exitCode'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                env=env, timeout=360).returncode
            row['after'] = B.environment()
            B.save(root / 'report.json', report)
            if row['exitCode']:
                raise RuntimeError(case['name'] + ': inference failed; see preserved log')
            new_config = json.loads((output / 'config.json').read_text())
            for key in ('photoIDs', 'sourceSHA256', 'datasetSHA256'):
                assert new_config[key] == old_config[key], key
            assert new_config['backendVersion'] == args.backend_version
            assert new_config['executionPolicy'] == args.execution_policy
            assert new_config['mpsFallback'] == '0'
            checks = {}
            with np.load(reference / 'predictions.npz') as a, np.load(output / 'predictions.npz') as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    checks[key] = a[key].dtype == b[key].dtype and bool(np.array_equal(a[key], b[key]))
            row['predictionArraysExactlyEqual'] = checks
            row['pixelTransformsExactlyEqual'] = (json.loads((reference / 'pixel-transforms.json').read_text()) ==
                                                  json.loads((output / 'pixel-transforms.json').read_text()))
            audit = json.loads((output / 'kernel-audit.json').read_text())
            assert audit['backendVersion'] == args.backend_version and audit['executionPolicy'] == args.execution_policy
            row['packedKernelCalls'] = audit['packedKernelCalls']
            row['executedLowBitParameterCount'] = audit['executedLowBitParameterCount']
            assert row['packedKernelCalls'] == 288
            assert row['executedLowBitParameterCount'] == new_config['weightFormat']['lowBitParameterCount']
            resources = [json.loads(line) for line in (output / 'resources.jsonl').read_text().splitlines()]
            physical = [r['physicalFootprintBytes'] for r in resources if r.get('physicalFootprintBytes') is not None]
            row['sampledPhysicalPeakBytes'] = max(physical) if physical else None
            row['referencesAndDatasetUnchanged'] = all(B.sha(p) == checksum for p, checksum in original.items())
            row['predictionsSHA256'] = B.sha(output / 'predictions.npz')
            B.save(root / 'report.json', report)
            assert all(checks.values()) and row['pixelTransformsExactlyEqual'] and row['referencesAndDatasetUnchanged']
            row['status'] = 'exact-prediction-parity-passed'
            B.save(root / 'report.json', report)
            print(json.dumps({key: row[key] for key in ('name', 'status', 'sampledPhysicalPeakBytes')}), flush=True)
        assert identities == {p.name: B.sha(p) for p in worker.glob('*.py')}
        report['status'] = 'packaged-worker-regressions-passed' if args.packaged_worker else 'worker-overlay-regressions-passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        B.save(root / 'report.json', report)


if __name__ == '__main__':
    main()
