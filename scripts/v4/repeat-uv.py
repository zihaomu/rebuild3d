#!/usr/bin/env python3
"""Interleave isolated, fresh UV runs on fixed A/B meshes and one native build.

This measures UV only. It does not satisfy the goal's full reconstruction,
application-click, GPU-memory or visual acceptance gates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def snapshot():
    result = {'atUTC': datetime.now(timezone.utc).isoformat(),
              'loadAverage': os.getloadavg(), 'platform': platform.platform()}
    for name, function in [('memory', psutil.virtual_memory), ('swap', psutil.swap_memory)]:
        try:
            result[name] = function()._asdict()
        except (OSError, psutil.Error) as error:
            result[name] = {'unavailable': str(error)}
    for name, command in [('power', ['pmset', '-g', 'batt']),
                          ('thermal', ['pmset', '-g', 'therm'])]:
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = {'exitCode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {'unavailable': str(error)}
    result['processInventory'] = 'not sampled; background-load matching is not proven'
    return result


def compare_uv(first, current):
    with np.load(first) as reference, np.load(current) as candidate:
        return {key: np.array_equal(reference[key], candidate[key])
                for key in reference.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--geometry-a', type=Path, required=True)
    parser.add_argument('--geometry-b', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 3:
        raise ValueError('At least three rounds are required')
    if args.extension.name != 'xatlas.cpython-312-darwin.so':
        raise ValueError('Expected the pinned CPython 3.12 xatlas extension')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = output / 'worker'
    shutil.copytree(ROOT / 'Runtime/rebuild3d_worker', worker / 'rebuild3d_worker',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    native = output / 'native'
    native.mkdir()
    shutil.copy2(args.extension, native / args.extension.name)
    geometries = {'A': args.geometry_a.resolve(), 'B': args.geometry_b.resolve()}
    identities = {name: sha(path) for name, path in geometries.items()}
    sources = {str(path.relative_to(output)): sha(path) for path in sorted(worker.rglob('*'))
               if path.is_file()}
    schedule = []
    for round_number in range(1, args.rounds + 1):
        # Reverse both dataset and treatment order in every other round.
        labels = ('A', 'B') if round_number % 2 else ('B', 'A')
        budgets = (0, 32768) if round_number % 2 else (32768, 0)
        for label in labels:
            for budget in budgets:
                schedule.append({'round': round_number, 'dataset': label, 'budget': budget})
    manifest = {'scope': 'UV-only serial experiment; not final full-task/application acceptance',
                'python': str(args.python.absolute()), 'scriptSHA256': sha(__file__),
                'geometrySHA256': identities, 'workerSourceSHA256': sources,
                'xatlasExtensionSHA256': sha(native / args.extension.name),
                'resolution': 2048, 'padding': 4, 'bilinear': True,
                'schedule': schedule, 'environmentBefore': snapshot(),
                'outputReuse': False, 'thermalControl': 'observe only; no power or OS changes'}
    save(output / 'protocol.json', manifest)
    child = None
    stopped = False

    def stop(_number, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join((str(native), str(worker))))
    results = []
    first_uv = {}
    for item in schedule:
        if stopped:
            return 130
        label, budget = item['dataset'], item['budget']
        if sha(geometries[label]) != identities[label]:
            raise ValueError('Geometry changed during the experiment')
        name = f"{label}-r{item['round']}-{'whole' if budget == 0 else 'partition32768'}"
        target = output / name
        command = [str(args.python.absolute()), '-m', 'rebuild3d_worker.unwrap',
                   str(geometries[label]), str(target), '--size', '2048',
                   '--chunk-face-budget', str(budget)]
        before = snapshot()
        print(json.dumps({'event': 'start', 'name': name, 'atUTC': before['atUTC']}), flush=True)
        start = time.monotonic()
        with (output / (name + '.log')).open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            save(output / 'process.json', {'name': name, 'pid': child.pid, 'command': command})
            code = child.wait()
        row = {**item, 'name': name, 'exitCode': code,
               'processWallSeconds': time.monotonic() - start, 'before': before, 'after': snapshot()}
        report_file = target / 'report.json'
        if code == 0 and report_file.is_file():
            report = json.loads(report_file.read_text())
            if (report['inputSHA256'] != identities[label]
                    or report['xatlasExtensionSHA256'] != manifest['xatlasExtensionSHA256']
                    or not all(report[key] for key in ('surfaceExactlyPreserved', 'sameFaceCornerOrder', 'allUVFacesHaveArea'))):
                raise ValueError('UV identity or surface invariant failed')
            row['report'] = report
            key = (label, budget)
            if key in first_uv:
                row['repeatArrayEquality'] = compare_uv(first_uv[key], target / 'uv.npz')
            else:
                first_uv[key] = target / 'uv.npz'
            row['uvSHA256'] = sha(target / 'uv.npz')
        results.append(row)
        save(output / 'samples.json', results)
        print(json.dumps({'event': 'completed', 'name': name, 'exitCode': code,
                          'processWallSeconds': row['processWallSeconds']}), flush=True)
        if code != 0 or 'report' not in row:
            return code or 1
    summary = {'scope': manifest['scope'], 'samples': 'samples.json', 'datasets': {}}
    for label in geometries:
        comparison = {}
        for budget, key in [(0, 'whole'), (32768, 'partition32768')]:
            rows = [row for row in results if row['dataset'] == label and row['budget'] == budget]
            comparison[key] = {'samples': [row['name'] for row in rows],
                               'processWallSeconds': [row['processWallSeconds'] for row in rows],
                               'medianProcessWallSeconds': statistics.median(row['processWallSeconds'] for row in rows),
                               'medianComputeChartsSeconds': statistics.median(row['report']['timingsSeconds']['computeChartsSeconds'] for row in rows),
                               'medianPackChartsSeconds': statistics.median(row['report']['timingsSeconds']['packChartsSeconds'] for row in rows),
                               'allRepeatedUVArraysIdentical': all(all(row.get('repeatArrayEquality', {}).values()) for row in rows)}
        comparison['medianUVProcessTimeReduction'] = 1 - comparison['partition32768']['medianProcessWallSeconds'] / comparison['whole']['medianProcessWallSeconds']
        summary['datasets'][label] = comparison
    summary['finalApplicationAcceptance'] = 'not evaluated'
    summary['environmentAfter'] = snapshot()
    save(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
