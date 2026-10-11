#!/usr/bin/env python3
"""Interleaved packaged inference diagnosis, independent of GUI availability.

Reuses only frozen prepared photographs. Each sample loads its own packaged
runtime and weights in a new process. This is NOT fresh end-to-end application
timing and cannot satisfy the real click-to-visible-model requirement.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time

import numpy as np

HELPER = Path(__file__).with_name('benchmark.py')
SPEC = importlib.util.spec_from_file_location('benchmark', HELPER)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ui-protocol', type=Path, required=True)
    parser.add_argument('--baseline-inputs-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = load(args.ui_protocol)
    runtimes = {label: Path(app['app']) / 'Contents/Resources/GenerationRuntime'
                for label, app in protocol['apps'].items()}
    for label, runtime in runtimes.items():
        assert B.sha(runtime / 'runtime.json') == protocol['apps'][label]['runtimeManifestSHA256']
        for f in load(runtime / 'runtime.json')['artifacts']:
            p = runtime / f['path']
            assert p.resolve().is_relative_to(runtime)
            assert p.stat().st_size == f['byteCount'] and B.sha(p) == f['sha256']
    inputs = {dataset: args.baseline_inputs_root.resolve() / f'{dataset}-fresh-01/job/inputs'
              for dataset in ('A', 'B')}
    input_hashes = {}
    for dataset, folder in inputs.items():
        rows = load(folder / 'dataset.json')['records']
        assert [r['sourceSHA256'] for r in rows] == [r['sha256'] for r in protocol['sources'][dataset]]
        assert len(rows) == 7
        input_hashes[dataset] = {str(p): B.sha(p) for p in folder.rglob('*') if p.is_file()}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(HELPER, out / HELPER.name)
    report = {'status': 'running', 'scope': __doc__, 'startedAtUTC': now(), 'records': [],
              'uiProtocolSHA256': B.sha(args.ui_protocol), 'apps': protocol['apps'],
              'order': [{k: r[k] for k in ('name', 'dataset', 'label', 'round')} for r in protocol['order']],
              'preparedInputsSHA256': input_hashes, 'scriptSHA256': B.sha(__file__),
              'helperSHA256': B.sha(HELPER), 'goalComplete': False, 'guiAcceptance': False,
              'limits': ['Prepared-input inference only; excludes prepare/fusion/UV/texture/package/display',
                         'Independent processes, available filesystem/kernel caches, components already installed',
                         'Both backends run with MPS CPU fallback disabled',
                         'All samples retained; environment, thermal, sleep, pressure and swap require review']}
    B.save(out / 'report.json', report)
    reference = {}
    try:
        for row in report['order']:
            folder = out / row['name']; folder.mkdir()
            runtime, dataset = runtimes[row['label']], inputs[row['dataset']]
            env = os.environ.copy()
            for key in ('PYTHONHOME', 'VIRTUAL_ENV', 'REBUILD3D_PARENT_PID',
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO'):
                env.pop(key, None)
            env.update(PYTHONPATH=str(runtime / 'worker'), PYTHONUNBUFFERED='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1')
            command = [str(runtime / 'python/bin/python3.12'), '-B', '-m', 'rebuild3d_worker.infer',
                       str(dataset), str(folder / 'inference'), '--repo', str(runtime / 'vggt'),
                       '--weights', str(runtime / 'models/vggt-1b.safetensors'), '--size', '518',
                       '--count', '7', '--max-memory-gib', '10.5', '--max-seconds', '300']
            before = B.environment()
            started = time.monotonic(); wall_started = time.time()
            resources = []
            with (folder / 'inference.log').open('w') as log, (folder / 'resources.jsonl').open('w') as stream:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                record = {**row, 'status': 'running', 'pid': process.pid, 'startedAtUTC': now(),
                          'command': command, 'before': before}
                report['records'].append(record); B.save(out / 'report.json', report)
                print(json.dumps({'started': row['name'], 'pid': process.pid}), flush=True)

                def stop(signum, frame):
                    raise InterruptedError('Diagnostic runner interrupted')

                signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
                try:
                    while process.poll() is None:
                        if time.monotonic() - started > 360:
                            raise TimeoutError('Diagnostic inference exceeded 360 seconds')
                        processes = B.sample_tree(process.pid)
                        values = [p['physicalFootprintBytes'] for p in processes]
                        resource = {'atUTC': now(), 'seconds': time.monotonic() - started,
                                    'processes': processes,
                                    'treePhysicalFootprintBytes': sum(values) if values and all(v is not None for v in values) else None,
                                    'memoryPressureLevel': B.memory_pressure(), 'swap': B.psutil.swap_memory()._asdict(),
                                    'availableMemoryBytes': B.psutil.virtual_memory().available}
                        resources.append(resource); stream.write(json.dumps(resource) + '\n'); stream.flush()
                        time.sleep(.25)
                    code = process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); process.wait()
            elapsed, wall = time.monotonic() - started, time.time() - wall_started
            record.update(exitCode=code, processSeconds=elapsed, wallSeconds=wall,
                          finishedAtUTC=now(), after=B.environment())
            assert code == 0, f'{row["name"]} failed; see inference.log'
            inference = load(folder / 'inference/report.json')
            assert inference['status'] == 'inference-complete' and inference['cameraCount'] == 7
            config = load(folder / 'inference/config.json')
            assert config['arguments']['size'] == 518 and config['mpsFallback'] == '0'
            assert config['sourceSHA256'] == [r['sha256'] for r in protocol['sources'][row['dataset']]]
            key = row['dataset'], row['label']
            predictions = folder / 'inference/predictions.npz'
            if key in reference:
                with np.load(reference[key]) as first, np.load(predictions) as current:
                    assert first.files == current.files and all(np.array_equal(first[k], current[k]) for k in first.files)
            else:
                reference[key] = predictions
            valid = [r['treePhysicalFootprintBytes'] for r in resources if r['treePhysicalFootprintBytes'] is not None]
            assert valid
            pressure = [r['memoryPressureLevel'] for r in resources]
            gaps = [b['seconds'] - a['seconds'] for a, b in zip(resources, resources[1:])]
            record.update(status='diagnostic-sample-complete', inferenceReport=inference,
                          peakPhysicalFootprintBytes=max(valid), physicalMissingSamples=len(resources)-len(valid),
                          peakMemoryPressureLevel=max(pressure) if all(v is not None for v in pressure) else None,
                          maximumSampleGapSeconds=max(gaps), wallMinusMonotonicSeconds=wall-elapsed,
                          swapUsedDeltaBytes=record['after']['swap']['used']-before['swap']['used'],
                          swapOutDeltaBytes=record['after']['swap']['sout']-before['swap']['sout'],
                          predictionsSHA256=B.sha(predictions), identicalToFirstSameBackend=True,
                          inferenceReportSHA256=B.sha(folder / 'inference/report.json'))
            B.save(folder / 'measurement.json', record)
            B.save(out / 'report.json', report)
            print(json.dumps({'completed': row['name'], 'seconds': elapsed, 'physicalPeakBytes': max(valid)}), flush=True)
        comparisons = {}
        for dataset in ('A', 'B'):
            medians = {label: {k: statistics.median(r[k] for r in report['records']
                       if r['dataset'] == dataset and r['label'] == label)
                       for k in ('processSeconds', 'peakPhysicalFootprintBytes', 'swapUsedDeltaBytes', 'swapOutDeltaBytes')}
                       for label in runtimes}
            comparisons[dataset] = {'medians': medians, 'diagnosticReductions': {
                k: 1 - medians['candidate'][k] / medians['baseline'][k]
                for k in ('processSeconds', 'peakPhysicalFootprintBytes')}}
        assert all(B.sha(p) == digest for files in input_hashes.values() for p, digest in files.items())
        report.update(status='twelve-inference-diagnostics-complete-review-required', comparisons=comparisons,
                      originalPreparedInputsUnchanged=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finishedAtUTC'] = now(); B.save(out / 'report.json', report)


if __name__ == '__main__':
    main()
