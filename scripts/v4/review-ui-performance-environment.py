#!/usr/bin/env python3
"""Extract environment evidence for completed actual-UI performance samples.

This supplements, and never rewrites, observer reports or numeric summaries.
It does not automatically approve environmental comparability or the Goal.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def read(path):
    return json.loads(path.read_text())


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def top_processes(snapshot):
    rows = []
    result = snapshot['processes']
    if result['exitCode']:
        return {'error': result}
    for line in result['stdout'].splitlines()[1:]:
        fields = line.split(None, 4)
        if len(fields) == 5:
            rows.append({'pid': int(fields[0]), 'parentPID': int(fields[1]),
                         'cpuPercent': float(fields[2]), 'rssKiB': int(fields[3]),
                         'command': fields[4]})
    return {'byCPU': sorted(rows, key=lambda r: r['cpuPercent'], reverse=True)[:12],
            'byRSS': sorted(rows, key=lambda r: r['rssKiB'], reverse=True)[:12]}


def sample(root, row):
    folder = root / 'observations' / row['name']
    report, ui = read(folder / 'report.json'), read(folder / 'ui-observations.json')
    assert report['status'] == 'observation-complete-review-required'
    visible = [o for o in ui['observations'] if o['state'] == 'visible'
               and o.get('texturedModelVisible')]
    start = stamp(ui['clickCall']['startedAtUTC'])
    finish = min(stamp(o['finishedAtUTC']) for o in visible)
    rows = [json.loads(line) for line in (folder / 'resources.jsonl').read_text().splitlines()]
    measured = [r for r in rows if start <= stamp(r['atUTC']) <= finish]
    preceding = [r for r in rows if stamp(r['atUTC']) <= start]
    assert len(measured) >= 2 and preceding, 'Click/display resource bounds missing'
    first, last = preceding[-1], measured[-1]
    gaps = [stamp(b['atUTC']) - stamp(a['atUTC']) for a, b in zip(measured, measured[1:])]
    assert min(gaps) > 0
    clock_drift = [(stamp(r['atUTC']) - stamp(measured[0]['atUTC'])) -
                   (r['secondsSinceArmed'] - measured[0]['secondsSinceArmed']) for r in measured]
    pressure = [r['memoryPressureLevel'] for r in measured]
    return {
        'name': row['name'], 'dataset': row['dataset'], 'label': row['label'], 'round': row['round'],
        'appPID': report['pid'], 'appProcessCreated': report['processCreated'],
        'clickStartedAtUTC': ui['clickCall']['startedAtUTC'],
        'firstVisibleByUTC': datetime.fromtimestamp(finish, timezone.utc).isoformat(),
        'observerFinishedAtUTC': report['finishedAtUTC'],
        'beforeAndAfter': {phase: {
            'atUTC': report[phase]['atUTC'], 'platform': report[phase]['platform'],
            'memoryBytes': report[phase]['memoryBytes'], 'cpuCount': report[phase]['cpuCount'],
            'hardware': report[phase]['hardware'], 'power': report[phase]['power'],
            'thermal': report[phase]['thermal'], 'loadAverage': report[phase]['loadAverage'],
            'topProcesses': top_processes(report[phase]),
        } for phase in ('before', 'after')},
        'clickToVisibleResourceWindow': {
            'sampleCount': len(measured), 'maxSampleGapSeconds': max(gaps),
            'firstSampleBeforeClickSeconds': start - stamp(first['atUTC']),
            'lastSampleBeforeVisibleBoundSeconds': finish - stamp(last['atUTC']),
            'maxAbsoluteWallMinusMonotonicDriftSeconds': max(map(abs, clock_drift)),
            'pressureLevels': sorted(set(pressure), key=lambda p: -1 if p is None else p),
            'peakPressureLevel': max(pressure) if all(p is not None for p in pressure) else None,
            'swapUsedDeltaBytes': last['swap']['used'] - first['swap']['used'],
            'swapOutDeltaBytes': last['swap']['sout'] - first['swap']['sout'],
            'swapInDeltaBytes': last['swap']['sin'] - first['swap']['sin'],
            'minimumAvailableMemoryBytes': min(r['availableMemoryBytes'] for r in measured),
            'systemCPUPercentMedian': statistics.median(r['systemCPUPercent'] for r in measured),
            'systemCPUPercentMaximum': max(r['systemCPUPercent'] for r in measured),
        },
        'evidenceSHA256': {p.name: sha(p) for p in (folder / 'report.json', folder / 'resources.jsonl',
                                                  folder / 'ui-observations.json')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path, help='New output directory; existing evidence is preserved')
    args = parser.parse_args()
    protocol = read(args.root / 'protocol.json')
    rows = [sample(args.root, row) for row in protocol['order']]
    assert len(rows) == 12
    assert len({(r['appPID'], r['appProcessCreated']) for r in rows}) == 12
    for previous, current in zip(rows, rows[1:]):
        assert stamp(previous['observerFinishedAtUTC']) <= stamp(current['clickStartedAtUTC'])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    result = {'status': 'environment-evidence-ready-for-review',
              'createdAtUTC': datetime.now(timezone.utc).isoformat(),
              'protocolSHA256': sha(args.root / 'protocol.json'), 'scriptSHA256': sha(Path(__file__)),
              'sampleCount': len(rows), 'distinctAppProcesses': 12, 'orderedNonoverlappingObservers': True,
              'samples': rows, 'environmentAccepted': None, 'goalComplete': False,
              'limits': [
                  'Resource window uses actual click and first visible observation bounds, not backend completion.',
                  'Raw observer before/after swap deltas remain unchanged; this separately bounds idle-time effects.',
                  'System pressure and swap include other applications and require comparative human review.',
                  'Process snapshots do not continuously measure unrelated GPU workloads.',
                  'pmset warning history is not a continuous temperature or clock-frequency measurement.',
                  'Clock drift and sample gaps are evidence, not automatic sleep or performance acceptance.',
              ]}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'sampleCount': len(rows)}))


if __name__ == '__main__':
    main()
