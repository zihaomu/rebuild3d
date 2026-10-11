#!/usr/bin/env python3
"""Compare a scheduling experiment with frozen candidate 09 in fresh processes.

This reuses prepared photographs and preimports worker modules in the probe.
It is a diagnostic, not full application timing or final Goal acceptance.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

HELPER = Path(__file__).with_name('benchmark.py')
SPEC = importlib.util.spec_from_file_location('benchmark', HELPER)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--inputs-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['async-blocks', 'batched-load', 'nonblocking-heads'], required=True)
    parser.add_argument('--rest-seconds', type=float, default=0,
                        help='Fixed idle interval before every sample; not a thermal guarantee')
    parser.add_argument('--state-probe', type=Path,
                        help='Optional compiled system-state.swift probe; records current thermal/low-power state')
    args = parser.parse_args()
    if not 0 <= args.rest_seconds <= 60:
        raise ValueError('Expected a fixed idle interval between 0 and 60 seconds')
    runtime = args.runtime.resolve()
    root = args.output.resolve()
    probe = Path(__file__).with_name('probe-inference-scheduling.py').resolve()
    references = {key: args.reference_root.resolve() / name / 'inference/predictions.npz'
                  for key, name in [('A', '02-A-candidate-r1'), ('B', '03-B-candidate-r1')]}
    order = [('A', 'control', 1), ('A', args.mode, 1), ('B', args.mode, 1), ('B', 'control', 1),
             ('A', args.mode, 2), ('A', 'control', 2), ('B', 'control', 2), ('B', args.mode, 2),
             ('A', 'control', 3), ('A', args.mode, 3), ('B', args.mode, 3), ('B', 'control', 3)]
    assert B.sha(runtime / 'runtime.json') == '1796bc7e6bff7fc2c59397fcef304268b2cf71c11915b54169425846cfb95069'
    root.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), HELPER, probe):
        shutil.copy2(path, root / path.name)
    report = {'status': 'running', 'scope': __doc__, 'mode': args.mode, 'order': order,
              'runtimeManifestSHA256': B.sha(runtime / 'runtime.json'),
              'scriptSHA256': B.sha(__file__), 'helperSHA256': B.sha(HELPER),
              'probeSHA256': B.sha(probe), 'records': [], 'goalComplete': False,
              'fixedRestSeconds': args.rest_seconds,
              'systemStateProbeSHA256': B.sha(args.state_probe) if args.state_probe else None,
              'referenceSHA256': {key: B.sha(value) for key, value in references.items()}}
    B.save(root / 'report.json', report)

    def environment():
        value = B.environment()
        if args.state_probe:
            value['currentSystemState'] = json.loads(subprocess.check_output(
                [str(args.state_probe.resolve())], text=True, timeout=10))
        return value

    try:
        for index, (dataset, mode, repeat) in enumerate(order, 1):
            name = f'{index:02d}-{dataset}-{mode}-r{repeat}'
            output = root / name
            inputs = args.inputs_root.resolve() / f'{dataset}-fresh-01/job/inputs'
            command = [str(runtime / 'python/bin/python3.12'), '-B', str(probe), str(inputs), str(output),
                       '--runtime', str(runtime), '--reference', str(references[dataset]), '--mode', mode]
            env = os.environ.copy()
            for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'REBUILD3D_PARENT_PID',
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO'):
                env.pop(key, None)
            env.update(PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1')
            print(json.dumps({'restBefore': name, 'seconds': args.rest_seconds}), flush=True)
            rest_started = time.monotonic()
            time.sleep(args.rest_seconds)
            row = {'name': name, 'dataset': dataset, 'mode': mode, 'round': repeat,
                   'command': command, 'actualRestSeconds': time.monotonic() - rest_started,
                   'before': environment()}
            report['records'].append(row)
            B.save(root / 'report.json', report)
            print(json.dumps({'started': name}), flush=True)
            started = time.monotonic()
            with (root / f'{name}.log').open('w') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=360)
            row.update(exitCode=result.returncode, processSeconds=time.monotonic() - started, after=environment())
            if (output / 'report.json').is_file():
                row['probe'] = json.loads((output / 'report.json').read_text())
            B.save(root / 'report.json', report)
            if result.returncode or row.get('probe', {}).get('status') != 'diagnostic-complete':
                raise RuntimeError(f'{name} failed or predictions changed; see preserved log/report')
            samples = [json.loads(line) for line in (output / 'inference/resources.jsonl').read_text().splitlines()]
            physical = [s['physicalFootprintBytes'] for s in samples if s.get('physicalFootprintBytes') is not None]
            if not physical:
                raise RuntimeError(f'{name} has no physical memory samples')
            row.update(peakPhysicalFootprintBytes=max(physical), physicalMissingSamples=len(samples) - len(physical),
                       internalInferenceSeconds=row['probe']['inferenceReport']['elapsedSeconds'],
                       timings=row['probe']['inferenceReport']['timingsSeconds'],
                       resourcesSHA256=B.sha(output / 'inference/resources.jsonl'),
                       probeReportSHA256=B.sha(output / 'report.json'))
            B.save(root / 'report.json', report)
            print(json.dumps({key: row[key] for key in ('name', 'processSeconds', 'peakPhysicalFootprintBytes', 'timings')}), flush=True)
        comparisons = {}
        for dataset in ('A', 'B'):
            medians = {}
            for mode in ('control', args.mode):
                rows = [r for r in report['records'] if r['dataset'] == dataset and r['mode'] == mode]
                medians[mode] = {key: statistics.median(r[key] for r in rows)
                                 for key in ('processSeconds', 'internalInferenceSeconds', 'peakPhysicalFootprintBytes')}
                medians[mode]['timings'] = {key: statistics.median(r['timings'][key] for r in rows)
                                          for key in rows[0]['timings']}
            comparisons[dataset] = {'medians': medians, 'diagnosticReductions': {
                key: 1 - medians[args.mode][key] / medians['control'][key]
                for key in ('processSeconds', 'internalInferenceSeconds', 'peakPhysicalFootprintBytes')}}
        assert B.sha(probe) == report['probeSHA256']
        assert all(B.sha(path) == report['referenceSHA256'][key] for key, path in references.items())
        report.update(status='twelve-scheduling-diagnostics-complete-review-required', comparisons=comparisons)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        B.save(root / 'report.json', report)


if __name__ == '__main__':
    main()
