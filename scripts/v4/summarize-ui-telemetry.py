#!/usr/bin/env python3
"""Summarize continuous telemetry without approving environment comparability.

The observer's top-CPU log contains application processes too. Exclude PIDs
actually observed in that sample's application tree before describing other
processes. The remaining top-15 sum is only a lower bound: inaccessible and
unlisted processes are not zero. No temperature or GPU-clock claim is made.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(folder):
    report = read(folder / 'report.json')
    ui = read(folder / 'ui-observations.json')
    assert report['status'] == 'observation-complete-review-required' and report['job']
    start = stamp(ui['clickCall']['startedAtUTC'])
    finish = min(stamp(x['finishedAtUTC']) for x in ui['observations']
                 if x['state'] == 'visible' and x.get('texturedModelVisible'))
    resources = lines(folder / 'resources.jsonl')
    thermal = lines(folder / 'system-state.jsonl')
    background = lines(folder / 'background-cpu.jsonl')
    own_pids = {p['pid'] for r in resources for p in r['appProcesses']}
    assert report['pid'] in own_pids
    active = [r for r in thermal if start <= stamp(r['atUTC']) <= finish]
    assert len(active) >= 2
    before = [r for r in thermal if stamp(r['atUTC']) <= start]
    assert before
    nominal = []
    for row in reversed(before):
        if row['thermalState'] != 'nominal' or row['lowPowerMode']:
            break
        nominal.append(row)
    nominal_seconds = (nominal[0]['systemUptimeSeconds'] - nominal[-1]['systemUptimeSeconds']
                       if nominal else 0)
    gaps = [stamp(b['atUTC']) - stamp(a['atUTC']) for a, b in zip(active, active[1:])]
    assert min(gaps) > 0
    other_totals, other, errors, excluded = [], {}, [], 0
    cpu_active = [r for r in background if start <= stamp(r['atUTC']) <= finish
                  and not r['warmingUp']]
    assert cpu_active
    for row in cpu_active:
        entries = row['topCPUProcesses']
        unowned = [p for p in entries if p['pid'] not in own_pids]
        excluded += len(entries) - len(unowned)
        other_totals.append(sum(p['cpuPercent'] for p in unowned))
        errors.append(row['observationErrors'])
        for process in unowned:
            key = (process['pid'], process['name'])
            other.setdefault(key, []).append(process['cpuPercent'])
    ranked = sorted(({'pid': key[0], 'name': key[1], 'listedSamples': len(values),
                      'maximumCPUPercent': max(values),
                      'medianCPUPercentWhenListed': statistics.median(values)}
                     for key, values in other.items()),
                    key=lambda x: x['maximumCPUPercent'], reverse=True)
    window = [r for r in resources if start <= stamp(r['atUTC']) <= finish]
    preceding = [r for r in resources if stamp(r['atUTC']) <= start]
    assert window and preceding
    first, last = preceding[-1], window[-1]
    return {
        'clickStartedAtUTC': ui['clickCall']['startedAtUTC'],
        'visibleByUTC': datetime.fromtimestamp(finish, timezone.utc).isoformat(),
        'preclickNominalConsecutiveSeconds': nominal_seconds,
        'preclickLastThermalGapSeconds': start - stamp(before[-1]['atUTC']),
        'thermal': {'samples': len(active),
                    'stateCounts': dict(Counter(r['thermalState'] for r in active)),
                    'lowPowerSampleCount': sum(r['lowPowerMode'] for r in active),
                    'maxSampleGapSeconds': max(gaps),
                    'lastSampleBeforeVisibleBoundSeconds': finish - stamp(active[-1]['atUTC'])},
        'otherObservedCPU': {
            'sampleCount': len(cpu_active), 'applicationPIDsExcluded': sorted(own_pids),
            'applicationEntriesRemoved': excluded,
            'medianListedCPUPercentSumLowerBound': statistics.median(other_totals),
            'maximumListedCPUPercentSumLowerBound': max(other_totals),
            'processReadErrorCountMinimum': min(errors),
            'processReadErrorCountMaximum': max(errors), 'highestObservedProcesses': ranked[:15]},
        'systemResourceWindow': {
            'pressureLevels': sorted({r['memoryPressureLevel'] for r in window}, key=str),
            'swapUsedDeltaBytes': last['swap']['used'] - first['swap']['used'],
            'swapOutDeltaBytes': last['swap']['sout'] - first['swap']['sout'],
            'swapInDeltaBytes': last['swap']['sin'] - first['swap']['sin']},
        'evidenceSHA256': {name: sha(folder / name) for name in (
            'report.json', 'ui-observations.json', 'resources.jsonl',
            'system-state.jsonl', 'background-cpu.jsonl', 'preclick-check.json')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path, help='New JSON file; never overwrite prior evidence')
    args = parser.parse_args()
    assert not args.output.exists()
    protocol = read(args.root / 'protocol.json')
    records = []
    for row in protocol['order']:
        folder = args.root / 'observations' / row['name']
        record = {key: row[key] for key in ('name', 'dataset', 'label', 'round')}
        report_path, ui_path = folder / 'report.json', folder / 'ui-observations.json'
        if not report_path.exists() or not ui_path.exists():
            record['status'] = 'not-ready'
        elif read(report_path)['status'] != 'observation-complete-review-required':
            record['status'] = 'observation-not-complete'
        else:
            record.update(summarize(folder), status='telemetry-summarized-review-required')
        records.append(record)
    result = {'status': 'telemetry-evidence-only',
              'createdAtUTC': datetime.now(timezone.utc).isoformat(),
              'scriptSHA256': sha(Path(__file__)),
              'protocolSHA256': sha(args.root / 'protocol.json'), 'samples': records,
              'summarizedSamples': sum(r['status'] == 'telemetry-summarized-review-required' for r in records),
              'environmentAccepted': None, 'goalComplete': False,
              'limits': [
                  'Top-15 logs are truncated and include inaccessible-process errors; sums are lower bounds.',
                  'Non-application processes include the measurement tools and task automation.',
                  'Application PIDs are excluded using observed resource samples; extremely short-lived processes or PID reuse cannot be fully resolved.',
                  'OS thermal classification is sampled, not temperature or GPU frequency.',
                  'Preclick conditions and in-run environment require review; no automatic performance acceptance.',
                  'System swap and pressure are shared with other applications.',
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'summarizedSamples': result['summarizedSamples']}))


if __name__ == '__main__':
    main()
