#!/usr/bin/env python3
"""Serial, interleaved fresh A/B tasks from frozen v3 and a packaged candidate.

Does not include GUI clicks, first installation, or display time. Every sample
starts a new worker and an empty task directory; all samples are retained.
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
from types import SimpleNamespace

HELPER = Path(__file__).with_name('benchmark.py')
SPEC = importlib.util.spec_from_file_location('benchmark', HELPER)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def load(path):
    return json.loads(path.read_text())


def summarize_sample(directory):
    measurement = load(directory / 'benchmark.json')
    events = [json.loads(line) for line in (directory / 'job/events.jsonl').read_text().splitlines()]
    started = {e['stage']: e for e in events if e['event'] == 'stage-started'}
    completed = {e['stage']: e for e in events if e['event'] == 'stage-completed'}
    assert measurement['exitCode'] == 0 and measurement['reusedStageCount'] == 0
    assert events[-1]['event'] == 'completed'
    samples = [json.loads(line) for line in (directory / 'process-resources.jsonl').read_text().splitlines()]
    start = datetime.fromisoformat(started['inference']['timestamp'])
    end = datetime.fromisoformat(completed['inference']['timestamp'])
    inference = [s for s in samples if start <= datetime.fromisoformat(s['atUTC']) <= end]
    assert inference and all(s['treePhysicalFootprintBytes'] is not None for s in inference)
    pressures = [s['memoryPressureLevel'] for s in samples if s['memoryPressureLevel'] is not None]
    return {'path': str(directory), 'backendSeconds': measurement['elapsedSeconds'],
            'inferenceSeconds': completed['inference']['stageElapsedSeconds'],
            'stageSeconds': {k: v['stageElapsedSeconds'] for k, v in completed.items()},
            'inferencePeakPhysicalFootprintBytes': max(s['treePhysicalFootprintBytes'] for s in inference),
            'wholeTaskPeakPhysicalFootprintBytes': measurement['sampledPeakTreePhysicalFootprintBytes'],
            'inferenceSamples': len(inference),
            'swapUsedDeltaBytes': measurement['after']['swap']['used'] - measurement['before']['swap']['used'],
            'swapOutDeltaBytes': measurement['after']['swap']['sout'] - measurement['before']['swap']['sout'],
            'peakMemoryPressureLevel': max(pressures) if pressures else None,
            'freshStageReuseCount': 0, 'measurementSHA256': B.sha(directory / 'benchmark.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path('build/v4-quantization/baseline/protocol.json'))
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtimes = {'baseline': a.baseline.resolve(), 'candidate': a.candidate.resolve()}
    protocol = load(a.protocol)
    assert B.sha(runtimes['baseline'] / 'runtime.json') == protocol['runtimeManifestSHA256']
    for runtime in runtimes.values():
        for f in load(runtime / 'runtime.json')['artifacts']:
            path = runtime / f['path']
            assert path.resolve().is_relative_to(runtime)
            assert path.stat().st_size == f['byteCount'] and B.sha(path) == f['sha256'], str(path)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(HELPER, out / HELPER.name)
    order = []
    for round_index in range(3):
        for dataset in ('A', 'B'):
            labels = ['baseline', 'candidate']
            if (round_index + (dataset == 'B')) % 2:
                labels.reverse()
            order.extend({'round': round_index + 1, 'dataset': dataset, 'label': label} for label in labels)
    manifests = {}
    for dataset in ('A', 'B'):
        original = load(a.protocol.parent / (dataset + '-task-template.json'))
        for photo in original['photos']:
            assert B.sha(photo['sourcePath']) == photo['sourceSHA256']
        for label, runtime in runtimes.items():
            task = json.loads(json.dumps(original))
            task['runtime'] = {'nativeWorker': str(runtime / 'rebuild3d-native-worker'),
                'rasterLibrary': str(runtime / 'worker/rebuild3d_worker/raster.dylib'),
                'vggtRepository': str(runtime / 'vggt'), 'weights': str(runtime / 'models/vggt-1b.safetensors')}
            filename = out / f'{dataset}-{label}-task.json'
            B.save(filename, task)
            manifests[dataset, label] = filename
    freeze = {'scope': __doc__, 'order': order, 'runtimeManifestSHA256': {
        k: B.sha(v / 'runtime.json') for k, v in runtimes.items()},
        'qualityProtocolSHA256': B.sha(a.protocol), 'helperSHA256': B.sha(HELPER),
        'sourceInputsSHA256': {str(k): B.sha(v) for k, v in manifests.items()},
        'comparison': 'Same 518 geometry, 256 volume and 4096 texture; no input-derived task cache. Components already available. No simultaneous reconstruction/archive/test jobs.',
        'before': B.environment()}
    B.save(out / 'freeze.json', freeze)
    report = {'status': 'running', 'scope': __doc__, 'records': [], 'freezeSHA256': B.sha(out / 'freeze.json')}
    B.save(out / 'report.json', report)
    try:
        for index, row in enumerate(order):
            dataset, label = row['dataset'], row['label']
            runtime = runtimes[label]
            assert B.sha(runtime / 'runtime.json') == freeze['runtimeManifestSHA256'][label]
            folder = out / f'{index + 1:02d}-{dataset}-{label}-r{row["round"]}'
            print(json.dumps({'sampleStarting': str(folder), **row}), flush=True)
            code = B.run(SimpleNamespace(output=folder, manifest=manifests[dataset, label],
                python=runtime / 'python/bin/python3.12', worker_root=runtime / 'worker', label=label))
            if code:
                raise RuntimeError(f'{folder.name} exited {code}; failed sample retained')
            report['records'].append({**row, **summarize_sample(folder)})
            B.save(out / 'report.json', report)
        summaries = {}
        for dataset in ('A', 'B'):
            medians = {}
            for label in runtimes:
                rows = [r for r in report['records'] if r['dataset'] == dataset and r['label'] == label]
                assert len(rows) == 3
                medians[label] = {key: statistics.median(r[key] for r in rows) for key in (
                    'backendSeconds', 'inferenceSeconds', 'inferencePeakPhysicalFootprintBytes',
                    'swapUsedDeltaBytes', 'swapOutDeltaBytes')}
                pressures = [r['peakMemoryPressureLevel'] for r in rows]
                medians[label]['maximumMemoryPressureLevel'] = max(pressures) if all(v is not None for v in pressures) else None
            reduction = {key: 1 - medians['candidate'][key] / medians['baseline'][key] for key in (
                'backendSeconds', 'inferenceSeconds', 'inferencePeakPhysicalFootprintBytes')}
            summaries[dataset] = {'medians': medians, 'reductions': reduction,
                'inferenceTimeGatePassed': reduction['inferenceSeconds'] >= .20,
                'inferenceMemoryGatePassed': reduction['inferencePeakPhysicalFootprintBytes'] >= .20,
                'swapAndPressureRequireMatchedEnvironmentReview': True}
        report.update(status='twelve-fresh-backend-samples-complete-environment-review-pending',
                      summaries=summaries, uiAcceptance=False, goalComplete=False)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        B.save(out / 'report.json', report)


if __name__ == '__main__':
    main()
