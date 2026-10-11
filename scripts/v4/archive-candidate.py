#!/usr/bin/env python3
"""Create and relocate a local candidate ZIP, then run packaged dependency checks.

Does not publish anything or claim offline application acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def logical_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source, output = args.app.resolve(), args.output.resolve()
    assert source.suffix == '.app' and (source / 'Contents/Resources/GenerationRuntime/runtime.json').is_file()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    zipped = output / (source.stem + '.zip')
    destination = output / 'relocated candidate 中文'
    relocated = destination / source.name
    manifest = 'Contents/Resources/GenerationRuntime/runtime.json'
    executable = 'Contents/MacOS/Rebuild3D'
    report = {'status': 'running', 'startedAtUTC': datetime.now(timezone.utc).isoformat(),
              'source': str(source), 'relocated': str(relocated), 'zip': str(zipped),
              'sourceExpandedLogicalBytes': logical_bytes(source),
              'runtimeManifestSHA256': sha(source / manifest), 'executableSHA256': sha(source / executable),
              'scriptSHA256': sha(Path(__file__)), 'processes': [],
              'limits': ['Local archive and dependency validation only; no UI/offline acceptance',
                         'Logical bytes are not APFS exclusive physical allocation or upgrade peak',
                         'Archive timings are diagnostic, not first preparation or generation performance']}

    def save():
        tmp = output / 'archive.tmp'
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(output / 'archive.json')

    def run(name, command):
        start = time.monotonic()
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = {'phase': name, 'command': command, 'exitCode': result.returncode,
                  'elapsedSecondsDiagnosticOnly': time.monotonic() - start}
        report['processes'].append(record)
        save()
        print(json.dumps({'phase': name, 'exitCode': result.returncode}), flush=True)
        assert result.returncode == 0, name

    save()
    try:
        run('archive', ['/usr/bin/ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', str(source), str(zipped)])
        report.update(zipBytes=zipped.stat().st_size, zipSHA256=sha(zipped))
        with zipfile.ZipFile(zipped) as archive:
            assert archive.testzip() is None
            assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in archive.namelist())
            report['zipMemberCount'] = len(archive.infolist())
        run('extract', ['/usr/bin/ditto', '-x', '-k', str(zipped), str(destination)])
        assert sha(source / manifest) == sha(relocated / manifest) == report['runtimeManifestSHA256']
        assert sha(source / executable) == sha(relocated / executable) == report['executableSHA256']
        report['relocatedExpandedLogicalBytes'] = logical_bytes(relocated)
        assert report['sourceExpandedLogicalBytes'] == report['relocatedExpandedLogicalBytes']
        probe = output / 'relocated.cpu-probe.json'
        run('cpu-probe', [sys.executable, '-B', str(Path(__file__).with_name('probe-runtime.py').resolve()),
                          str(relocated / 'Contents/Resources/GenerationRuntime'), str(probe)])
        audit = output / 'relocated.dependency-audit.json'
        run('dependency-audit', [sys.executable, '-B', str(Path(__file__).with_name('audit-app.py').resolve()),
                                str(relocated), str(audit), '--runtime-probe', str(probe)])
        report.update(status='local-archive-relocation-and-dependencies-passed',
                      cpuProbeSHA256=sha(probe), dependencyAuditSHA256=sha(audit))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finishedAtUTC'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
