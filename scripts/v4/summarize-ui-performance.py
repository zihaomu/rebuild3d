#!/usr/bin/env python3
"""Summarize frozen real-UI samples without hiding failed or missing runs.

Observer outputs belong in ROOT/observations/SAMPLE_NAME. Each requires an
operator-written ui-observations.json containing clickCall and observations.
Each UI call has startedAtUTC/finishedAtUTC; observations additionally have
state (visible/not-visible/error), evidence, and texturedModelVisible for visible.
Timing bounds include observation uncertainty. Environment acceptance remains
a separate review; this tool never declares the whole Goal complete.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return Path(path).read_text()


def data(path):
    return json.loads(load(path))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp(value):
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    assert date.tzinfo is not None, 'UTC/offset timestamps required'
    return date.timestamp()


def call_bounds(call):
    lo, hi = stamp(call['startedAtUTC']), stamp(call['finishedAtUTC'])
    assert lo <= hi, 'Reversed call timestamps'
    return lo, hi


def waiting_bounds(ui, completed_at):
    click_lo, click_hi = call_bounds(ui['clickCall'])
    visible = []
    absent = []
    for observation in ui['observations']:
        lo, hi = call_bounds(observation)
        if observation['state'] == 'visible':
            assert observation.get('texturedModelVisible') is True
            assert observation.get('evidence'), 'Actual visual observation required'
            visible.append((lo, hi))
        elif observation['state'] == 'not-visible':
            assert observation.get('evidence')
            absent.append((lo, hi))
    assert visible, 'No actual model-visible observation'
    first = min(visible, key=lambda x: x[1])
    lower_display = max([stamp(completed_at)] + [lo for lo, hi in absent if hi <= first[0]])
    assert click_lo <= lower_display <= first[1], 'UI and worker timelines disagree'
    return {'lowerSeconds': max(0, lower_display - click_hi),
            'upperSeconds': first[1] - click_lo,
            'observationWindowSeconds': first[1] - lower_display,
            'basis': 'Lower: backend completion or last confirmed absence. Upper: first actual visible observation. Click call is also an interval.'}


def peak(rows, key):
    values = [r.get(key) for r in rows]
    assert values and all(v is not None for v in values), 'Missing physical-memory samples'
    return max(values)


def summarize_sample(root, row, protocol):
    folder = root / 'observations' / row['name']
    report = data(folder / 'report.json')
    assert report['status'] == 'observation-complete-review-required', report['status']
    app = protocol['apps'][row['label']]
    for key in ('app', 'binarySHA256', 'runtimeManifestSHA256'):
        assert report[key] == app[key], 'App identity mismatch: ' + key
    assert report['project'] == row['project']
    assert report['projectBeforeSHA256'] == row['projectManifestSHA256']
    assert report['photoFilesSHA256'] == row['photoFilesSHA256']
    assert report['originalPhotosUnchanged'] and report['stageReuseCount'] == 0
    project, job = Path(row['project']), Path(report['job'])
    assert list((project / 'generation').glob('*/request.json')) == [job / 'request.json'], 'Retry or extra job'
    for path, digest in row['photoFilesSHA256'].items():
        assert sha(project / path) == digest
    request = data(job / 'request.json')
    expected = protocol['fixedParameters']
    assert len(request['photos']) == expected['photoCount'] == 7
    assert all(request['parameters'][k] == expected[k] for k in ('modelSize', 'textureSize', 'volumeResolution'))
    config = data(job / 'inference/config.json')
    assert config['arguments']['size'] == expected['modelSize']
    assert config['sourceSHA256'] == [p['sha256'] for p in protocol['sources'][row['dataset']]]
    events = [json.loads(line) for line in load(job / 'events.jsonl').splitlines()]
    assert events[-1]['event'] == 'completed'
    assert not any(e['event'] in ('stage-reused', 'failed', 'cancelled') for e in events)
    starts = [e for e in events if e['event'] == 'stage-started']
    ends = [e for e in events if e['event'] == 'stage-completed']
    stages = ['prepare', 'inference', 'fusion', 'unwrap', 'atlas', 'texture', 'package']
    assert [e['stage'] for e in starts] == [e['stage'] for e in ends] == stages
    start, end = stamp(starts[1]['timestamp']), stamp(ends[1]['timestamp'])
    resources = [json.loads(line) for line in load(folder / 'resources.jsonl').splitlines()]
    inference = [r for r in resources if start <= stamp(r['atUTC']) <= end]
    infer_report = data(job / 'inference/report.json')
    assert infer_report['status'] == 'inference-complete' and infer_report['cameraCount'] == 7
    native = [json.loads(line) for line in load(job / 'inference/resources.jsonl').splitlines()]
    native_peaks = {k: max(r[k] for r in native if k in r) for k in
                    ('mpsDriverBytes', 'mlxActiveBytes', 'mlxCacheBytes', 'mlxPeakActiveBytes')
                    if any(k in r for r in native)}
    intervals = [stamp(b['atUTC']) - stamp(a['atUTC']) for a, b in zip(resources, resources[1:])]
    assert intervals and min(intervals) >= 0
    pressures = [r['memoryPressureLevel'] for r in resources]
    ui = data(folder / 'ui-observations.json')
    assert stamp(report['armedAtUTC']) <= call_bounds(ui['clickCall'])[0]
    timings = infer_report.get('timingsSeconds')
    network_keys = ('aggregator', 'cameraHeadAndReadback', 'depthHeadAndReadback')
    network = sum(timings[k] for k in network_keys) if timings and all(k in timings for k in network_keys) else None
    return {'status': 'sample-verified-environment-review-pending', 'name': row['name'],
            'dataset': row['dataset'], 'label': row['label'], 'round': row['round'],
            'waitingTimeBounds': waiting_bounds(ui, events[-1]['timestamp']),
            'inferenceSeconds': ends[1]['stageElapsedSeconds'],
            'stageSeconds': {e['stage']: e['stageElapsedSeconds'] for e in ends},
            'backendSeconds': events[-1]['elapsedSeconds'],
            'inferencePeakPhysicalFootprintBytes': peak(inference, 'workerTreePhysicalFootprintBytes'),
            'wholeAppSampledPeakPhysicalFootprintBytes': peak(resources, 'appTreePhysicalFootprintBytes'),
            'inferenceSampleCount': len(inference), 'samplingIntervalMedianSeconds': statistics.median(intervals),
            'samplingIntervalMaxSeconds': max(intervals), 'nativeMemoryPeaksDoNotAdd': native_peaks,
            'inferenceSubstageSeconds': timings, 'networkWithHeadReadbackSeconds': network,
            'substageLimits': 'Includes per-process first execution/compilation; not isolated steady-state kernels. Legacy baseline has no substage timestamps; missing values remain null and require supplemental profiling.',
            'swapUsedDeltaBytes': report['after']['swap']['used'] - report['before']['swap']['used'],
            'swapOutDeltaBytes': report['after']['swap']['sout'] - report['before']['swap']['sout'],
            'peakMemoryPressureLevel': max(pressures) if all(p is not None for p in pressures) else None,
            'evidenceSHA256': {str(p): sha(p) for p in
                (folder / 'report.json', folder / 'resources.jsonl', folder / 'ui-observations.json',
                 job / 'events.jsonl', job / 'inference/report.json', job / 'bundle/model.usdz')}}


def compare(rows):
    result = {}
    for dataset in ('A', 'B'):
        groups = {label: [r for r in rows if r['dataset'] == dataset and r['label'] == label]
                  for label in ('baseline', 'candidate')}
        if any(len(g) != 3 or {r['round'] for r in g} != {1, 2, 3} for g in groups.values()):
            result[dataset] = {'status': 'incomplete', 'gates': None}
            continue
        medians = {}
        for label, samples in groups.items():
            medians[label] = {key: statistics.median(s[key] for s in samples) for key in
                             ('inferenceSeconds', 'inferencePeakPhysicalFootprintBytes',
                              'swapUsedDeltaBytes', 'swapOutDeltaBytes')}
            medians[label]['waitingBounds'] = {bound: statistics.median(s['waitingTimeBounds'][bound] for s in samples)
                                               for bound in ('lowerSeconds', 'upperSeconds')}
        baseline, candidate = medians['baseline'], medians['candidate']
        reductions = {k: 1 - candidate[k] / baseline[k] for k in
                      ('inferenceSeconds', 'inferencePeakPhysicalFootprintBytes')}
        b, c = baseline['waitingBounds'], candidate['waitingBounds']
        conservative = 1 - c['upperSeconds'] / b['lowerSeconds'] if b['lowerSeconds'] > 0 else None
        optimistic = 1 - c['lowerSeconds'] / b['upperSeconds']
        wait_gate = 'passed' if conservative is not None and conservative >= .15 else (
            'failed' if optimistic < .15 else 'inconclusive-observation-uncertainty')
        result[dataset] = {'status': 'numeric-summary-environment-review-pending', 'medians': medians,
                           'reductions': reductions,
                           'waitingReductionBounds': {'lower': conservative, 'upper': optimistic},
                           'gates': {'inferenceTime': reductions['inferenceSeconds'] >= .20,
                                     'inferenceMemory': reductions['inferencePeakPhysicalFootprintBytes'] >= .20,
                                     'actualUIWaiting': wait_gate},
                           'environmentPressureSwapSleepReview': 'required'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path, help='New JSON path; existing reports are preserved')
    args = parser.parse_args()
    protocol = data(args.root / 'protocol.json')
    samples = []
    for row in protocol['order']:
        try:
            sample = summarize_sample(args.root, row, protocol)
        except (OSError, ValueError, AssertionError, KeyError) as error:
            sample = {**{k: row[k] for k in ('name', 'dataset', 'label', 'round')},
                      'status': 'missing-or-unverified', 'error': repr(error)}
        samples.append(sample)
    valid = [r for r in samples if r['status'] == 'sample-verified-environment-review-pending']
    report = {'status': 'summary-environment-review-pending' if len(valid) == 12 else 'incomplete',
              'createdAtUTC': datetime.now(timezone.utc).isoformat(), 'protocolSHA256': sha(args.root / 'protocol.json'),
              'scriptSHA256': sha(Path(__file__)), 'samples': samples, 'verifiedSamples': len(valid),
              'comparisons': compare(valid), 'goalComplete': False}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'verifiedSamples': len(valid)}))


if __name__ == '__main__':
    main()
