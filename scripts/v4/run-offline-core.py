#!/usr/bin/env python3
"""Enforce offline execution for the production coordinator and packaged workers.

Process-scoped sandbox denies all network access and developer Python/source/
weight directories. This is core offline evidence; actual app UI is separate.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import threading
import time

SPEC = importlib.util.spec_from_file_location('matrix', Path(__file__).with_name('run-input-matrix.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
H = M.H


def load(path):
    return json.loads(path.read_text())


def totals(root):
    logical = blocks = partial = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            try:
                s = path.stat()
            except FileNotFoundError:
                continue
            logical += s.st_size
            blocks += s.st_blocks * 512
            if name.endswith('.partial'):
                partial += s.st_size
    return {'logicalBytes': logical, 'reportedAllocatedBlocksBytes': blocks, 'partialLogicalBytes': partial}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--app', type=Path, required=True)
    p.add_argument('--source-project', type=Path, required=True)
    p.add_argument('--import-probe', type=Path, required=True)
    p.add_argument('--generation-probe', type=Path, required=True)
    args = p.parse_args()
    out, app, source = [p.resolve() for p in (args.output, args.app, args.source_project)]
    seed = app / 'Contents/Resources/GenerationRuntime'
    repo = Path(__file__).resolve().parents[2]
    probe, importer = args.generation_probe.resolve(), args.import_probe.resolve()
    for executable in [probe, importer]:
        build = load(executable.parent / 'build.json')
        assert H.sha(executable) == build['binarySHA256']
        assert all(H.sha(path) == digest for path, digest in build['sources'].items())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    original = H.snapshot(source)
    manifest = load(source / 'project.json')
    sources = [{'path': str(source / photo['imagePath']), 'sha256': photo['metadata']['contentSHA256']}
               for photo in manifest['photos']]
    # The importer records original filenames from copied source URLs.
    inputs = out / 'inputs'; inputs.mkdir()
    for entry, photo in zip(sources, manifest['photos']):
        path = inputs / photo['originalName']
        subprocess.run(['/bin/cp', '-c', entry['path'], str(path)], check=True)
        assert H.sha(path) == entry['sha256']; entry['path'] = str(path)
    H.save(out / 'inputs.json', {'sources': sources})
    project = out / 'offline-new.rebuild3d'
    with (out / 'import.log').open('w') as log:
        imported = subprocess.run([str(importer), str(out / 'inputs.json'), str(project)], stdout=log, stderr=subprocess.STDOUT)
    assert imported.returncode == 0
    denied = [repo / 'Runtime', repo / 'build/v4-quantization/mlx-env', repo / 'build/v4-quantization/weights']
    profile = out / 'offline.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n' +
                       ''.join('(deny file-read* (subpath ' + json.dumps(str(path)) + '))\n' for path in denied))
    wrapper = out / 'coordinator-offline'
    wrapper.write_text('#!/bin/sh\nexec /usr/bin/sandbox-exec -f ' + shlex.quote(str(profile)) + ' ' + shlex.quote(str(probe)) + ' "$@"\n')
    wrapper.chmod(0o755)
    report = {'status': 'running', 'scope': __doc__, 'startedAtUTC': H.now(),
              'runtimeManifestSHA256': H.sha(seed / 'runtime.json'), 'app': str(app),
              'sourceProject': str(source), 'deniedDeveloperDirectories': list(map(str, denied)),
              'sandboxProfileSHA256': H.sha(profile), 'wrapperSHA256': H.sha(wrapper),
              'appBytes': totals(app), 'scriptSHA256': H.sha(__file__),
              'limits': ['Not actual GUI/offline button acceptance', 'Preparation time measured with occupancy sampler; not formal performance',
                         'Allocated blocks can share APFS extents and are not exclusive disk use',
                         '0.5 second sampling can miss transient disk peaks; raw samples retained']}
    H.save(out / 'report.json', report)
    stop = threading.Event()
    monitor = None
    try:
        # Positive loopback control and the exact same child under network denial.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen(2)
            code = 'import socket,sys; s=socket.socket(); s.settimeout(5); s.connect(("127.0.0.1",int(sys.argv[1]))); print("connected")'
            command = [str(seed / 'python/bin/python3.12'), '-I', '-B', '-c', code, str(listener.getsockname()[1])]
            control = subprocess.run(command, capture_output=True, text=True)
            blocked = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), *command], capture_output=True, text=True)
        report['networkControl'] = {'controlExitCode': control.returncode, 'controlStdout': control.stdout,
                                     'restrictedExitCode': blocked.returncode, 'restrictedStderr': blocked.stderr}
        assert control.returncode == 0 and blocked.returncode != 0 and 'Operation not permitted' in blocked.stderr
        store = out / 'runtime-store'
        samples = []
        def observe():
            with (out / 'disk-samples.jsonl').open('w') as stream:
                while not stop.is_set():
                    record = {'atUTC': H.now(), **totals(store)}
                    stream.write(json.dumps(record) + '\n'); stream.flush(); samples.append(record)
                    stop.wait(.5)
        monitor = threading.Thread(target=observe, daemon=True); monitor.start()
        result = H.run_phase(wrapper, seed, store, project, out / 'generated', None)
        assert result['exitCode'] == 0
        record = M.validate_job(project, sources)
        report['jobChecks'] = record
        command = [str(seed / 'python/bin/python3.12'), '-I', '-B', str(Path(__file__).with_name('audit-application-export.py').resolve()),
                   str(project), str(out / 'generated/recovered.usdz'), str(out / 'export-audit')]
        with (out / 'export-audit.log').open('w') as log:
            audit = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), *command], stdout=log, stderr=subprocess.STDOUT,
                                   env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'})
        assert audit.returncode == 0
        stop.set(); monitor.join()
        report['componentStoreFinal'] = totals(store)
        report['componentStoreObservedPeakLogicalBytes'] = max([r['logicalBytes'] for r in samples] + [totals(store)['logicalBytes']])
        report['freshAppAndComponentStoreLogicalBytes'] = report['appBytes']['logicalBytes'] + totals(store)['logicalBytes']
        report['exportAuditSHA256'] = H.sha(out / 'export-audit/report.json')
        report['status'] = 'network-denied-packaged-core-generation-and-export-passed'
    except BaseException as e:
        report.update(status='failed', error=repr(e)); raise
    finally:
        stop.set()
        if monitor: monitor.join()
        report['sourceUnchanged'] = original == H.snapshot(source)
        report['finishedAtUTC'] = H.now()
        H.save(out / 'report.json', report)
        assert report['sourceUnchanged']


if __name__ == '__main__':
    main()
