#!/usr/bin/env python3
"""Audit a local candidate's files and native link paths without launching UI.

The report combines static paths with an optional isolated CPU load probe. It
cannot prove GPU execution, one-click generation or offline app acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import re
import subprocess
import sys

MACH_MAGICS = {b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
               b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def commands(path):
    result = subprocess.run(['otool', '-l', str(path)], capture_output=True, text=True, check=True)
    parsed = []
    for block in re.split(r'Load command \d+\n', result.stdout)[1:]:
        command = re.search(r'^\s*cmd (\S+)', block, re.M)
        if not command:
            continue
        kind = command.group(1)
        if kind == 'LC_RPATH':
            parsed.append((kind, re.search(r'^\s*path (.+) \(offset', block, re.M).group(1)))
        elif kind in {'LC_ID_DYLIB', 'LC_LOAD_DYLIB', 'LC_LOAD_WEAK_DYLIB', 'LC_REEXPORT_DYLIB', 'LC_LOAD_UPWARD_DYLIB'}:
            parsed.append((kind, re.search(r'^\s*name (.+) \(offset', block, re.M).group(1)))
    return parsed


def is_system(path):
    return path.startswith(('/System/Library/', '/usr/lib/'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--runtime-probe', type=Path,
                        help='Optional isolated CPU probe proving actual Python extension load order')
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Report must be new')
    app = args.app.resolve()
    runtime = app / 'Contents/Resources/GenerationRuntime'
    manifest = json.loads((runtime / 'runtime.json').read_text())
    artifact_paths = set()
    for item in manifest['artifacts']:
        path = runtime / item['path']
        if item['path'] in artifact_paths or not path.resolve().is_relative_to(runtime):
            raise ValueError('Duplicate or escaping runtime path: ' + item['path'])
        artifact_paths.add(item['path'])
        if not path.is_file() or path.stat().st_size != item['byteCount'] or sha(path) != item['sha256']:
            raise ValueError('Changed/missing runtime artifact: ' + item['path'])
    actual = {str(path.relative_to(runtime)) for path in runtime.rglob('*') if path.is_file()}
    if actual != artifact_paths | {'runtime.json'}:
        raise ValueError('Unexpected/unmanifested runtime files: ' + repr(actual ^ (artifact_paths | {'runtime.json'})))
    files = sorted(path for path in app.rglob('*') if path.is_file())
    if any(path.is_symlink() for path in app.rglob('*')):
        raise ValueError('Candidate must have no external symlink dependencies')
    binaries = {}
    for path in files:
        with path.open('rb') as stream:
            if stream.read(4) in MACH_MAGICS:
                binaries[path] = commands(path)
    # @executable_path is resolved for the actual owning interpreter/native app,
    # never to a developer Python installation.
    interpreter = runtime / 'python/bin/python3.12'
    app_binary = app / 'Contents/MacOS/Rebuild3D'
    native_worker = runtime / 'rebuild3d-native-worker'
    unresolved, external, edges = [], [], []
    executable_for = {}
    for loader, loads in binaries.items():
        if loader.is_relative_to(runtime / 'python') or loader.is_relative_to(runtime / 'worker'):
            executable = interpreter
        elif loader == native_worker:
            executable = native_worker
        else:
            executable = app_binary
        executable_for[loader] = executable
        rpaths = [(loader, value) for kind, value in loads if kind == 'LC_RPATH']
        if executable != loader:
            rpaths += [(executable, value) for kind, value in binaries.get(executable, []) if kind == 'LC_RPATH']

        def expand(path, owner):
            return path.replace('@loader_path', str(owner.parent)).replace('@executable_path', str(executable.parent))

        for kind, dependency in loads:
            if kind in {'LC_RPATH', 'LC_ID_DYLIB'}:
                continue
            row = {'loader': str(loader.relative_to(app)), 'kind': kind, 'dependency': dependency}
            if is_system(dependency):
                row['resolution'] = 'system shared cache/framework; symbols not loaded by this audit'
            else:
                if dependency.startswith('@rpath/'):
                    suffix = dependency[len('@rpath/'):]
                    candidates = [Path(expand(base, owner)) / suffix for owner, base in rpaths]
                else:
                    candidates = [Path(expand(dependency, loader))]
                match = next((path.resolve() for path in candidates if path.is_absolute() and path.is_file()), None)
                if match is not None and match.is_relative_to(app):
                    row['resolution'] = str(match.relative_to(app))
                elif match is not None and is_system(str(match)):
                    row['resolution'] = 'system'
                elif dependency.startswith('/') and not is_system(dependency):
                    row['resolution'] = 'external absolute dependency'
                    external.append(row)
                else:
                    # An inherited dyld rpath can require runtime proof. Keep
                    # this unknown instead of inventing a resolved path.
                    row['resolution'] = 'unresolved by direct loader/interpreter rpaths'
                    unresolved.append(row)
            edges.append(row)

    # LC_RPATH is inherited along the actual loading chain. For example,
    # mlx/core -> libmlx -> libjaccl uses the core extension's rpath.
    parents = {path: [] for path in binaries}
    for row in edges:
        target = app / row['resolution']
        if target in parents:
            parents[target].append(app / row['loader'])

    def inherited_rpaths(loader, seen):
        if loader in seen:
            return []
        seen = seen | {loader}
        found = [(loader, value) for kind, value in binaries.get(loader, []) if kind == 'LC_RPATH']
        for parent in parents.get(loader, []):
            found.extend(inherited_rpaths(parent, seen))
        return found

    still_unresolved = []
    for row in unresolved:
        loader = app / row['loader']
        dependency = row['dependency']
        match = None
        if dependency.startswith('@rpath/'):
            for owner, rpath in inherited_rpaths(loader, set()):
                base = rpath.replace('@loader_path', str(owner.parent)).replace('@executable_path', str(executable_for[loader].parent))
                path = Path(base) / dependency[len('@rpath/'):]
                if path.is_absolute() and path.is_file() and path.resolve().is_relative_to(app):
                    match = path.resolve()
                    break
        if match:
            row.update(resolution=str(match.relative_to(app)), evidence='inherited loader-chain LC_RPATH')
        else:
            still_unresolved.append(row)
    unresolved = still_unresolved
    static_unresolved = len(unresolved)

    if args.runtime_probe:
        probe = json.loads(args.runtime_probe.read_text())
        if (probe.get('exitCode') != 0 or probe['runtimeManifestSHA256'] != sha(runtime / 'runtime.json')
                or not probe['probe'].get('cpuSmokeChecksPassed')
                or probe['probe'].get('externalNonSystemLibraries') != []):
            raise ValueError('CPU probe does not verify this exact runtime')
        loaded = {Path(path).resolve() for path in probe['probe']['loadedMachOImages']}
        still_unresolved = []
        for row in unresolved:
            loader = app / row['loader']
            matches = [path for path in loaded if path in binaries
                       and ('LC_ID_DYLIB', row['dependency']) in binaries[path]]
            # Torchvision loads after Torch. Require proof that this extension
            # and its install-name dependency were both actually loaded.
            if loader in loaded and len(matches) == 1:
                row.update(resolution=str(matches[0].relative_to(app)), evidence='actual isolated CPU process; matching loaded install name')
            else:
                still_unresolved.append(row)
        unresolved = still_unresolved
    info = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    signature = subprocess.run(['codesign', '--verify', '--strict', str(app)], capture_output=True, text=True)
    result = {'scope': 'Local candidate integrity/link-path audit with optional CPU load proof; not application acceptance',
              'app': str(app), 'bundleID': info['CFBundleIdentifier'],
              'expandedBytes': sum(path.stat().st_size for path in files),
              'runtimeManifestSHA256': sha(runtime / 'runtime.json'),
              'verifiedRuntimeArtifacts': len(artifact_paths), 'machOFiles': len(binaries),
              'codeSignature': {'exitCode': signature.returncode, 'stderr': signature.stderr},
              'nativeDependencies': edges, 'externalDependencies': external,
              'unresolvedDependencies': unresolved,
              'dependenciesRequiringRuntimeLoadProof': static_unresolved,
              'runtimeProbeSHA256': sha(args.runtime_probe) if args.runtime_probe else None,
              'integrityAndDependencyChecksPassed': not external and not unresolved and signature.returncode == 0,
              'applicationAcceptance': 'pending; launch/Metal/dynamic imports/offline/save/export not tested'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'nativeDependencies'}, indent=2))
    return 0 if result['integrityAndDependencyChecksPassed'] else 1


if __name__ == '__main__':
    sys.exit(main())
