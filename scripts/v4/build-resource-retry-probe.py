#!/usr/bin/env python3
"""Build a disclosed test-only launcher that forwards two MPS allocator limits.

Production coordinator retry logic and the packaged runtime remain unchanged.
The generated probe is not the production app binary or UI acceptance evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    copied = output / 'core'
    copied.mkdir()
    sources = sorted((root / 'Sources/Rebuild3DCore').glob('*.swift'))
    originals = {p.name: sha(p) for p in sources}
    for source in sources:
        shutil.copy2(source, copied / source.name)
    target = copied / 'GenerationCoordinator.swift'
    before = target.read_text()
    anchor = '                process.standardOutput = pipe; process.standardError = pipe\n'
    addition = '''                // Test-only launch transport; retry decisions are unchanged.
                for key in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"] {
                    if let value = ProcessInfo.processInfo.environment[key] {
                        process.environment?[key] = value
                    }
                }
'''
    assert before.count(anchor) == 1
    after = before.replace(anchor, addition + anchor)
    assert after.replace(addition, '') == before
    target.write_text(after)
    probe = Path(__file__).with_name('probe-generation-recovery.swift')
    shutil.copy2(probe, output / probe.name)
    shutil.copy2(__file__, output / Path(__file__).name)
    command = ['swiftc', '-parse-as-library', '-O', '-swift-version', '6',
               *[str(copied / p.name) for p in sources], str(output / probe.name),
               '-o', str(output / 'probe-resource-retry')]
    with (output / 'compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report = {'scope': 'Test-only environment forwarding at process launch; not an unmodified app binary',
              'sourceRoot': str(root), 'originalCoreSHA256': originals,
              'compiledCoreSHA256': {p.name: sha(copied / p.name) for p in sources},
              'onlyModification': {'file': target.name, 'insertedBefore': anchor, 'insertedText': addition},
              'probeSourceSHA256': sha(probe), 'buildScriptSHA256': sha(__file__),
              'command': command, 'exitCode': result.returncode,
              'productionSourcesUnchanged': all(sha(p) == originals[p.name] for p in sources),
              'limits': ['No changed retry, identity, cancellation or publication logic',
                         'No application GUI or real system-wide low-memory claim']}
    if result.returncode == 0:
        report['binarySHA256'] = sha(output / 'probe-resource-retry')
    (output / 'build.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    assert report['productionSourcesUnchanged']
    print(json.dumps({'exitCode': result.returncode, 'output': str(output)}), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
