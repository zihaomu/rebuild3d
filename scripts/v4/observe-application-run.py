#!/usr/bin/env python3
"""Observe an already-open application while an operator uses its real UI.

Start after importing a fresh project and before clicking Generate. Stop by
writing stop.json in the output folder after recording actual UI observations.
This script never clicks, starts a reconstruction, edits a project, or kills it.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil

HELPER = Path(__file__).with_name('benchmark.py')
SPEC = importlib.util.spec_from_file_location('benchmark', HELPER)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('project', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--app', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--system-state-probe', type=Path,
                   help='Optional compiled system-state.swift; sample thermal classification every second')
    a = p.parse_args()
    project, out, app = a.project.resolve(), a.output.resolve(), a.app.resolve()
    process = psutil.Process(a.pid)
    assert Path(process.exe()).resolve() == app / 'Contents/MacOS/Rebuild3D'
    created = process.create_time()
    assert not list((project / 'generation').glob('*/request.json')), 'Fresh project required'
    manifest = json.loads((project / 'project.json').read_text())
    assert manifest['photos'] and not manifest.get('model'), 'Imported, ungenerated project required'
    photos = {p['imagePath']: B.sha(project / p['imagePath']) for p in manifest['photos']}
    runtime = app / 'Contents/Resources/GenerationRuntime'
    identity = B.sha(runtime / 'runtime.json')
    installed = Path.home() / 'Library/Application Support/org.rebuild3d.app/Runtimes' / identity
    assert (installed / 'runtime.json').exists() and B.sha(installed / 'runtime.json') == identity, 'Prepare components before steady-state timing'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(HELPER, out / HELPER.name)
    report = {'status': 'observing', 'scope': __doc__, 'armedAtUTC': now(),
        'project': str(project), 'app': str(app), 'pid': a.pid, 'processCreated': created,
        'runtimeManifestSHA256': identity, 'binarySHA256': B.sha(app / 'Contents/MacOS/Rebuild3D'),
        'projectBeforeSHA256': B.sha(project / 'project.json'), 'photoFilesSHA256': photos,
        'samplingIntervalSeconds': .25, 'before': B.environment(), 'uiAcceptance': False,
        'physicalMetric': 'libproc process-tree physical footprint; separate from RSS and GPU allocation'}
    thermal_child = None
    thermal_log = None
    if a.system_state_probe:
        probe = a.system_state_probe.resolve()
        report['systemStateProbe'] = {'path': str(probe), 'sha256': B.sha(probe),
                                     'intervalSeconds': 1, 'output': 'system-state.jsonl',
                                     'scope': 'Sampled OS thermal classification, not GPU clocks or temperature'}
        thermal_log = (out / 'system-state.jsonl').open('w')
        thermal_child = subprocess.Popen([str(probe), '--interval-seconds', '1'],
                                         stdout=thermal_log, stderr=subprocess.STDOUT)
    B.save(out / 'report.json', report)
    print('READY ' + str(out), flush=True)
    started = time.monotonic()
    job = None
    events = []
    worker = None
    worker_created = None
    background_previous = {}
    background_at = time.monotonic()
    background_samples = 0
    try:
        with (out / 'resources.jsonl').open('w') as stream, \
                (out / 'background-cpu.jsonl').open('w') as background:
            while not (out / 'stop.json').exists():
                if time.monotonic() - started > a.timeout:
                    raise TimeoutError('Observer timeout; application left untouched')
                if process.create_time() != created or not process.is_running():
                    raise RuntimeError('Observed app exited or changed PID identity')
                if thermal_child is not None and thermal_child.poll() is not None:
                    raise RuntimeError('Thermal state probe exited before observation completed')
                # Record competing CPU activity throughout the run, rather than
                # inferring matched conditions from two endpoint snapshots.
                sampled_at = time.monotonic()
                if sampled_at - background_at >= 2 or background_samples == 0:
                    current, rows = {}, []
                    errors = 0
                    for item in psutil.process_iter(['pid', 'name', 'create_time', 'cpu_times']):
                        try:
                            info = item.info
                            identity = (info['pid'], info['create_time'])
                            cpu = info['cpu_times'].user + info['cpu_times'].system
                            current[identity] = cpu
                            if identity in background_previous:
                                rows.append({'pid': info['pid'], 'name': info['name'],
                                             'cpuPercent': max(0, (cpu - background_previous[identity])
                                                               / (sampled_at - background_at) * 100)})
                        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                            errors += 1
                    background.write(json.dumps({'atUTC': now(), 'intervalSeconds': sampled_at - background_at,
                        'topCPUProcesses': sorted(rows, key=lambda x: x['cpuPercent'], reverse=True)[:15],
                        'observationErrors': errors, 'warmingUp': background_samples == 0}) + '\n')
                    background.flush()
                    background_previous, background_at = current, sampled_at
                    background_samples += 1
                jobs = list((project / 'generation').glob('*/request.json'))
                if jobs:
                    assert len(jobs) == 1, 'Unexpected retry must be reviewed separately'
                    job = jobs[0].parent
                    event_path = job / 'events.jsonl'
                    if event_path.exists():
                        lines = event_path.read_text().splitlines()
                        events = []
                        for line in lines:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                break  # Writer may be completing the last line.
                if worker is None:
                    for child in process.children(recursive=True):
                        try:
                            command = child.cmdline()
                            if 'rebuild3d_worker.worker' in command and any(str(project) in part for part in command):
                                worker, worker_created = child.pid, child.create_time()
                                report['workerCommand'] = command
                                report['workerPID'] = worker
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                app_rows = B.sample_tree(a.pid)
                worker_rows = []
                if worker is not None:
                    try:
                        if psutil.Process(worker).create_time() == worker_created:
                            worker_rows = B.sample_tree(worker)
                    except psutil.NoSuchProcess:
                        pass
                def total(rows):
                    return sum(r['physicalFootprintBytes'] for r in rows) if rows and all(r['physicalFootprintBytes'] is not None for r in rows) else None
                row = {'atUTC': now(), 'secondsSinceArmed': time.monotonic() - started,
                    'stage': events[-1].get('stage') if events else None,
                    'lastEvent': events[-1]['event'] if events else None,
                    'appTreePhysicalFootprintBytes': total(app_rows),
                    'workerTreePhysicalFootprintBytes': total(worker_rows),
                    'appProcesses': app_rows, 'workerProcesses': worker_rows,
                    'memoryPressureLevel': B.memory_pressure(), 'swap': psutil.swap_memory()._asdict(),
                    'availableMemoryBytes': psutil.virtual_memory().available,
                    'systemCPUPercent': psutil.cpu_percent()}
                stream.write(json.dumps(row) + '\n'); stream.flush()
                time.sleep(.25)
        assert photos == {p: B.sha(project / p) for p in photos}
        report.update(status='observation-complete-review-required', job=str(job) if job else None,
                      stageEvents=[e for e in events if e['event'] != 'diagnostic'],
                      stageReuseCount=sum(e['event'] == 'stage-reused' for e in events),
                      stop=json.loads((out / 'stop.json').read_text()), originalPhotosUnchanged=True)
    except BaseException as error:
        report.update(status='observation-incomplete', error=repr(error))
        raise
    finally:
        if thermal_child is not None:
            was_running = thermal_child.poll() is None
            if was_running:
                thermal_child.terminate()
            try:
                thermal_child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                thermal_child.kill()
                thermal_child.wait()
            thermal_log.close()
            report['systemStateProbe'].update(runningBeforeStop=was_running,
                                              exitCode=thermal_child.returncode)
        report.update(finishedAtUTC=now(), after=B.environment())
        B.save(out / 'report.json', report)


if __name__ == '__main__':
    main()
