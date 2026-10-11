#!/usr/bin/env python3
"""Prepare a process-scoped offline launcher for an unchanged candidate app.

Preparation runs only network controls. Launch the resulting test .app through
the normal UI automation entry point, then operate the candidate's real window.
The launcher records its PID before exec; verify that the actual app has that
same PID before counting any GUI observations as offline evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import shlex
import socket
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--app', required=True, type=Path)
    args = parser.parse_args()
    out, app = args.output.resolve(), args.app.resolve()
    binary = app / 'Contents/MacOS/Rebuild3D'
    runtime = app / 'Contents/Resources/GenerationRuntime'
    python = runtime / 'python/bin/python3.12'
    assert binary.is_file() and python.is_file()
    repo = Path(__file__).resolve().parents[2]
    denied = [repo / 'Runtime', repo / 'build/v4-quantization/mlx-env',
              repo / 'build/v4-quantization/weights']
    out.mkdir(parents=True, exist_ok=False)
    profile = out / 'offline.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n' + ''.join(
        '(deny file-read* (subpath ' + json.dumps(str(path)) + '))\n' for path in denied))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(2)
        port = listener.getsockname()[1]
        code = ('import socket,sys; s=socket.socket(); s.settimeout(5); '
                's.connect(("127.0.0.1",int(sys.argv[1]))); print("connected")')
        command = [str(python), '-I', '-B', '-c', code, str(port)]
        control = subprocess.run(command, capture_output=True, text=True)
        restricted = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), *command],
                                    capture_output=True, text=True)
    assert control.returncode == 0
    assert restricted.returncode != 0 and 'Operation not permitted' in restricted.stderr
    config = {'app': str(app), 'binary': str(binary), 'binarySHA256': sha(binary),
              'runtimeManifestSHA256': sha(runtime / 'runtime.json'),
              'loopbackPort': port,
              'deniedProbeFile': str(repo / 'Runtime/rebuild3d_worker/infer.py')}
    (out / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    bootstrap = out / 'bootstrap.py'
    bootstrap.write_text('''import errno, hashlib, json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(__file__).resolve().parent
config = json.loads((root / 'configuration.json').read_text())
record = {'status': 'checking-restrictions', 'atUTC': datetime.now(timezone.utc).isoformat(),
          'pid': os.getpid(), 'configuration': config}
def save():
    (root / 'launch-evidence.json').write_text(json.dumps(record, indent=2) + '\\n')
save()
try:
    with socket.socket() as connection:
        connection.settimeout(5)
        connection.connect(('127.0.0.1', config['loopbackPort']))
except OSError as error:
    record['networkError'] = {'errno': error.errno, 'message': str(error)}
    assert error.errno in (errno.EPERM, errno.EACCES), 'Network denial not established'
else:
    raise RuntimeError('Unexpected successful network connection')
try:
    with open(config['deniedProbeFile'], 'rb') as stream:
        stream.read(1)
except OSError as error:
    record['developerSourceError'] = {'errno': error.errno, 'message': str(error)}
    assert error.errno in (errno.EPERM, errno.EACCES)
else:
    raise RuntimeError('Developer source was readable')
with open(config['binary'], 'rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == config['binarySHA256']
record['status'] = 'restrictions-confirmed-exec-pending'
save()
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0')
os.execve(config['binary'], [config['binary']], env)
''')
    launcher = out / 'Rebuild3D-Offline-Test.app'
    macos = launcher / 'Contents/MacOS'
    macos.mkdir(parents=True)
    # LaunchServices retains the launcher's identity across exec. Match the
    # unchanged application's identity so accessibility can resolve its window.
    # Only one candidate process may be running during this isolated test.
    app_info = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    info = {'CFBundleIdentifier': app_info['CFBundleIdentifier'],
            'CFBundleName': 'Rebuild3D Offline Test', 'CFBundleExecutable': 'launch',
            'CFBundlePackageType': 'APPL', 'CFBundleVersion': '1',
            'LSMinimumSystemVersion': '15.0'}
    (launcher / 'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    executable = macos / 'launch'
    executable.write_text('#!/bin/sh\nexec ' + shlex.join([
        '/usr/bin/sandbox-exec', '-f', str(profile), str(python), '-I', '-B', str(bootstrap)]) +
        ' > ' + shlex.quote(str(out / 'application.log')) + ' 2>&1\n')
    executable.chmod(0o755)
    subprocess.run(['/usr/bin/codesign', '--force', '--sign', '-', str(launcher)], check=True)
    report = {'status': 'prepared-not-launched', 'scope': __doc__, 'app': str(app),
              'launcher': str(launcher), 'configuration': config,
              'scriptSHA256': sha(Path(__file__)), 'profileSHA256': sha(profile),
              'bootstrapSHA256': sha(bootstrap),
              'positiveControl': {'exitCode': control.returncode, 'stdout': control.stdout},
              'restrictedControl': {'exitCode': restricted.returncode, 'stderr': restricted.stderr},
              'candidateUnmodified': sha(binary) == config['binarySHA256'],
              'guiAcceptance': False,
              'limits': ['Process-scoped restrictions; host networking unchanged',
                         'Preparation does not prove GUI generation or display',
                         'Verify exec PID and actual window before using this as offline evidence']}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(launcher)


if __name__ == '__main__':
    main()
