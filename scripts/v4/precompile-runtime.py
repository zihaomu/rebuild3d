#!/usr/bin/env python3
"""Build source-checked bytecode before a new runtime manifest is sealed.

Run with that runtime's CPython using -I -B. Sources and licenses are retained;
optimization level zero retains assertions and docstrings. The final runtime
manifest must also hash every generated .pyc and this bytecode manifest.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compile_sources(root, sources):
    """Compile explicit sources to adjacent caches, rejecting foreign paths."""
    root = root.resolve()
    if sys.pycache_prefix is not None:
        raise ValueError('External Python cache prefix is not allowed in a shipped runtime')
    entries = []
    for source in sorted(sources):
        if source.is_symlink() or not source.is_file() or source.suffix != '.py':
            raise ValueError('Expected a regular Python source: ' + str(source))
        source = source.resolve()
        if not source.is_relative_to(root):
            raise ValueError('Source is outside the new runtime')
        relative = source.relative_to(root)
        target = Path(importlib.util.cache_from_source(str(source), optimization=''))
        if not target.resolve().is_relative_to(root) or target.exists() or target.is_symlink():
            raise ValueError('Bytecode destination is foreign or already exists')
        source_sha = sha(source)
        # Avoid embedding a developer username/build directory in the archive.
        py_compile.compile(str(source), cfile=str(target),
                           dfile='rebuild3d-runtime/' + relative.as_posix(),
                           doraise=True, optimize=0,
                           invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
        data = target.read_bytes()
        if (data[:4] != importlib.util.MAGIC_NUMBER or int.from_bytes(data[4:8], 'little') != 3
                or data[8:16] != importlib.util.source_hash(source.read_bytes()) or sha(source) != source_sha):
            raise ValueError('Bytecode/source validation failed: ' + str(relative))
        entries.append({'source': relative.as_posix(), 'sourceSHA256': source_sha,
                        'cache': target.relative_to(root).as_posix(),
                        'byteCount': len(data), 'sha256': sha(target)})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    root = args.runtime.resolve()
    if (root / 'runtime.json').exists() or (root / 'bytecode-manifest.json').exists():
        raise ValueError('Refuse to modify a sealed or already compiled runtime')
    if (sys.version_info[:2] != (3, 12) or sys.flags.optimize != 0
            or Path(sys.executable).resolve() != (root / 'python/bin/python3.12').resolve()):
        raise ValueError('Compile with the unoptimized packaged CPython 3.12 interpreter')
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('Build requires -I -B to isolate imports and automatic cache writes')
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths) or any(p.suffix == '.pyc' for p in paths):
        raise ValueError('New runtime must contain no symlinks or existing bytecode')
    sources = [p for p in paths if p.is_file() and p.suffix == '.py']
    if not sources:
        raise ValueError('No runtime Python sources found')
    entries = compile_sources(root, sources)
    report = {'formatVersion': 1, 'policy': 'cpython312-checked-hash-opt0-v1',
              'pythonVersion': sys.version, 'cacheTag': sys.implementation.cache_tag,
              'magicHex': importlib.util.MAGIC_NUMBER.hex(), 'optimize': 0,
              'invalidation': 'CHECKED_HASH', 'compilerSHA256': sha(Path(__file__)),
              'sourceFilesRetained': True, 'compiledFiles': len(entries),
              'byteCount': sum(e['byteCount'] for e in entries), 'entries': entries}
    (root / 'bytecode-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('policy', 'compiledFiles', 'byteCount')}))


if __name__ == '__main__':
    main()
