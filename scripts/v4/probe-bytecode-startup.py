#!/usr/bin/env python3
"""Diagnose checked-hash bytecode without changing a frozen runtime.

Compile with the runtime's own Python, then compare new inference processes
with/without an external bytecode directory. Prepared photos are reused; this
is not application performance acceptance or a deployable runtime package.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shutil
import signal
import statistics
import subprocess
import sys
import time

RUNTIME_SHA = '5d0446777eafc261caf583f386e634c013f330ad262ad345ad15a18790e82a10'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def compile_cache(args):
    runtime, out = args.runtime.resolve(), args.output.resolve()
    assert Path(sys.executable).resolve() == (runtime / 'python/bin/python3.12').resolve()
    assert sha(runtime / 'runtime.json') == RUNTIME_SHA
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    sys.pycache_prefix = str(out / 'cache')
    report = {'status': 'building', 'runtime': str(runtime), 'runtimeManifestSHA256': RUNTIME_SHA,
              'pythonVersion': sys.version, 'scriptSHA256': sha(__file__), 'createdAtUTC': now(),
              'invalidation': 'CHECKED_HASH', 'optimize': 0, 'compiled': [], 'failed': [],
              'scope': __doc__}
    for entry in read(runtime / 'runtime.json')['artifacts']:
        source = runtime / entry['path']
        if source.suffix != '.py':
            continue
        assert source.resolve().is_relative_to(runtime) and not source.is_symlink()
        assert sha(source) == entry['sha256']
        target = Path(importlib.util.cache_from_source(str(source), optimization=''))
        assert target.is_relative_to(out / 'cache')
        try:
            py_compile.compile(str(source), cfile=str(target), doraise=True, optimize=0,
                               invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
        except py_compile.PyCompileError as error:
            report['failed'].append({'source': entry['path'], 'error': str(error)})
            continue
        header = target.read_bytes()[:16]
        assert header[:4] == importlib.util.MAGIC_NUMBER
        assert int.from_bytes(header[4:8], 'little') == 3
        assert header[8:16] == importlib.util.source_hash(source.read_bytes())
        report['compiled'].append({'source': entry['path'], 'sourceSHA256': entry['sha256'],
                                   'cache': str(target.relative_to(out)), 'bytes': target.stat().st_size,
                                   'sha256': sha(target)})
    report.update(status='cache-built-diagnostic-only', totalBytes=sum(e['bytes'] for e in report['compiled']))
    save(out / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'totalBytes', 'failed')}), flush=True)
    print(json.dumps({'compiledFiles': len(report['compiled'])}), flush=True)


def screen(args):
    import numpy as np
    helper = Path(__file__).with_name('benchmark.py')
    spec = importlib.util.spec_from_file_location('bytecode_benchmark', helper)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    runtime, out, cache = args.runtime.resolve(), args.output.resolve(), args.cache.resolve()
    assert sha(runtime / 'runtime.json') == RUNTIME_SHA
    built = read(cache / 'report.json')
    assert built['runtimeManifestSHA256'] == RUNTIME_SHA and built['runtime'] == str(runtime)
    for entry in built['compiled']:
        assert sha(runtime / entry['source']) == entry['sourceSHA256']
        assert sha(cache / entry['cache']) == entry['sha256']
    inputs, references = {}, {}
    for dataset, name in [('A', '10-A-candidate-r3'), ('B', '11-B-candidate-r3')]:
        job = Path(read(args.ui_root / 'observations' / name / 'report.json')['job'])
        inputs[dataset], references[dataset] = job / 'inputs', job / 'inference/predictions.npz'
        assert len(read(inputs[dataset] / 'dataset.json')['records']) == 7
    order = [('A', 'control'), ('A', 'bytecode'), ('B', 'bytecode'), ('B', 'control')]
    if args.rounds == 3:
        order += [('A', 'bytecode'), ('A', 'control'), ('B', 'control'), ('B', 'bytecode'),
                  ('A', 'control'), ('A', 'bytecode'), ('B', 'bytecode'), ('B', 'control')]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(helper, out / helper.name)
    frozen_inputs = {str(p): sha(p) for folder in inputs.values() for p in folder.rglob('*') if p.is_file()}
    report = {'status': 'running', 'scope': __doc__, 'startedAtUTC': now(),
              'scriptSHA256': sha(__file__), 'helperSHA256': sha(helper),
              'runtimeManifestSHA256': RUNTIME_SHA, 'cacheReportSHA256': sha(cache / 'report.json'),
              'order': order, 'fixedRestSeconds': 30, 'records': [], 'goalComplete': False,
              'inputSHA256': frozen_inputs,
              'referenceSHA256': {k: sha(v) for k, v in references.items()}}
    save(out / 'report.json', report)

    def environment():
        return {**B.environment(), 'systemState': json.loads(subprocess.check_output(
            [str(args.state_probe.resolve())], text=True, timeout=10))}

    def stop(signum, frame):
        raise InterruptedError('Bytecode diagnostic interrupted')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for index, (dataset, mode) in enumerate(order, 1):
            folder = out / f'{index:02d}-{dataset}-{mode}'
            folder.mkdir()
            print(json.dumps({'restBefore': folder.name, 'seconds': 30}), flush=True)
            time.sleep(30)
            env = os.environ.copy()
            for key in ('PYTHONHOME', 'VIRTUAL_ENV', 'REBUILD3D_PARENT_PID', 'PYTHONOPTIMIZE',
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO'):
                env.pop(key, None)
            env.update(PYTHONPATH=str(runtime / 'worker'), PYTHONDONTWRITEBYTECODE='1',
                       PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1',
                       PYTHONPYCACHEPREFIX=str(cache / 'cache' if mode == 'bytecode' else out / 'empty-cache'))
            command = [str(runtime / 'python/bin/python3.12'), '-B', '-m', 'rebuild3d_worker.infer',
                       str(inputs[dataset]), str(folder / 'inference'), '--repo', str(runtime / 'vggt'),
                       '--weights', str(runtime / 'models/vggt-1b.safetensors'), '--count', '7', '--size', '518',
                       '--max-memory-gib', '10.5', '--max-seconds', '300']
            row = {'name': folder.name, 'dataset': dataset, 'mode': mode, 'before': environment(),
                   'command': command, 'pycachePrefix': env['PYTHONPYCACHEPREFIX'], 'startedAtUTC': now()}
            report['records'].append(row)
            save(out / 'report.json', report)
            started = time.monotonic()
            with (folder / 'inference.log').open('w') as log, (folder / 'resources.jsonl').open('w') as stream:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = process.pid
                samples = []
                try:
                    while process.poll() is None:
                        if time.monotonic() - started > 360:
                            raise TimeoutError('Inference exceeded 360 seconds')
                        tree = B.sample_tree(process.pid)
                        values = [p['physicalFootprintBytes'] for p in tree]
                        record = {'atUTC': now(), 'seconds': time.monotonic() - started,
                                  'treePhysicalFootprintBytes': sum(values) if values and all(v is not None for v in values) else None,
                                  'memoryPressureLevel': B.memory_pressure(), 'swap': B.psutil.swap_memory()._asdict()}
                        samples.append(record)
                        stream.write(json.dumps(record) + '\n'); stream.flush()
                        time.sleep(.25)
                    row.update(exitCode=process.wait(), processSeconds=time.monotonic() - started)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); process.wait()
            row.update(after=environment(), finishedAtUTC=now())
            assert row['exitCode'] == 0, f'{folder.name} failed; preserve log'
            row['inferenceReport'] = read(folder / 'inference/report.json')
            row['peakPhysicalFootprintBytes'] = max(v['treePhysicalFootprintBytes'] for v in samples if v['treePhysicalFootprintBytes'] is not None)
            with np.load(references[dataset]) as ref, np.load(folder / 'inference/predictions.npz') as result:
                assert ref.files == result.files
                row['arrayChecks'] = {key: ref[key].dtype == result[key].dtype and bool(np.array_equal(ref[key], result[key])) for key in ref.files}
            assert all(row['arrayChecks'].values()), 'Predictions differ'
            assert read(references[dataset].with_name('pixel-transforms.json')) == read(folder / 'inference/pixel-transforms.json')
            row['pixelTransformsExactlyEqual'] = True
            row['status'] = 'diagnostic-sample-complete'
            save(out / 'report.json', report)
            print(json.dumps({'name': folder.name, 'processSeconds': row['processSeconds'],
                              'peakPhysicalFootprintBytes': row['peakPhysicalFootprintBytes'],
                              'timings': row['inferenceReport']['timingsSeconds'], 'exactArrays': True}), flush=True)
        assert not (out / 'empty-cache').exists()
        assert all(sha(Path(p)) == h for p, h in frozen_inputs.items())
        assert all(sha(p) == report['referenceSHA256'][k] for k, p in references.items())
        assert all(sha(cache / e['cache']) == e['sha256'] for e in built['compiled'])
        comparisons = {}
        for dataset in ('A', 'B'):
            comparisons[dataset] = {}
            for mode in ('control', 'bytecode'):
                rows = [r for r in report['records'] if r['dataset'] == dataset and r['mode'] == mode]
                comparisons[dataset][mode] = {k: statistics.median(r[k] for r in rows) for k in ('processSeconds', 'peakPhysicalFootprintBytes')}
                comparisons[dataset][mode]['timings'] = {k: statistics.median(r['inferenceReport']['timingsSeconds'][k] for r in rows) for k in rows[0]['inferenceReport']['timingsSeconds']}
        report.update(status='diagnostic-complete-review-required', comparisons=comparisons,
                      originalInputsReferencesAndCacheUnchanged=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save(out / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['compile', 'screen'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--ui-root', type=Path)
    parser.add_argument('--state-probe', type=Path)
    parser.add_argument('--rounds', type=int, choices=[1, 3], default=1)
    args = parser.parse_args()
    if args.action == 'compile':
        compile_cache(args)
    else:
        if not all((args.cache, args.ui_root, args.state_probe)):
            parser.error('screen requires --cache, --ui-root and --state-probe')
        screen(args)


if __name__ == '__main__':
    main()
