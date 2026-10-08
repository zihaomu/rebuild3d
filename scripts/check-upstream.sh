#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
python3 - <<'PY'
from pathlib import Path
import platform
import subprocess

sources = sorted(str(path) for path in Path('Vendor/Photogrammetry/Photogrammetry').rglob('*.swift'))
command = ['xcrun', 'swiftc', '-swift-version', '5', '-target',
           f'{platform.machine()}-apple-macos26.0', '-o', 'build/Photogrammetry-upstream', *sources]
with open('build/upstream-build.log', 'w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
print(Path('build/upstream-build.log').read_text())
print(f'Upstream source compilation: {len(sources)} files, exit {result.returncode}')
raise SystemExit(result.returncode)
PY
