#!/usr/bin/env python3
"""Verify portable, source-checked cache behavior using actual subprocess imports."""
import importlib.util
import json
import marshal
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('runtime_bytecode', Path(__file__).with_name('precompile-runtime.py'))
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class RuntimeBytecodeTests(unittest.TestCase):
    def test_portable_cache_preserves_checks_and_rejects_stale_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'original'
            root.mkdir()
            source = root / 'fixture.py'
            source.write_text('"""Retained documentation."""\nVALUE = 7\ndef checked():\n    assert False, "retained assertion"\n')
            entry = B.compile_sources(root, [source])[0]
            cache = root / entry['cache']
            header = cache.read_bytes()[:16]
            self.assertEqual(int.from_bytes(header[4:8], 'little'), 3)
            code = marshal.loads(cache.read_bytes()[16:])
            self.assertEqual(code.co_filename, 'rebuild3d-runtime/fixture.py')
            moved = Path(temporary) / '搬移 folder'
            shutil.copytree(root, moved)
            original_cache = (moved / entry['cache']).read_bytes()
            probe = '''import sys,json
sys.path.insert(0,sys.argv[1])
from importlib.machinery import SourceFileLoader
def forbid_compile(*a,**k): raise RuntimeError("unexpected source compilation")
SourceFileLoader.source_to_code = forbid_compile
import fixture
try: fixture.checked()
except AssertionError: assertion = True
else: assertion = False
print(json.dumps([fixture.VALUE,fixture.__doc__,assertion]))
'''
            result = subprocess.check_output([sys.executable, '-I', '-B', '-c', probe, str(moved)], text=True)
            self.assertEqual(json.loads(result), [7, 'Retained documentation.', True])
            # Same timestamp and length, different content: checked-hash must reject the old cache.
            changed = moved / 'fixture.py'
            metadata = changed.stat()
            changed.write_text(changed.read_text().replace('VALUE = 7', 'VALUE = 8'))
            os.utime(changed, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            stale = subprocess.run([sys.executable, '-I', '-B', '-c', probe, str(moved)], capture_output=True, text=True)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn('unexpected source compilation', stale.stderr)
            code = 'import sys;sys.path.insert(0,sys.argv[1]);import fixture;print(fixture.VALUE)'
            result = subprocess.check_output([sys.executable, '-I', '-B', '-c', code, str(moved)], text=True)
            self.assertEqual(result.strip(), '8')
            self.assertEqual((moved / entry['cache']).read_bytes(), original_cache)

    def test_reproducible_across_build_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            blobs = []
            for name in ('first', 'second'):
                root = Path(temporary) / name
                root.mkdir()
                source = root / 'fixture.py'
                source.write_text('VALUE = 7\n')
                entry = B.compile_sources(root, [source])[0]
                blobs.append((root / entry['cache']).read_bytes())
                with self.assertRaises(ValueError):
                    B.compile_sources(root, [source])
            self.assertEqual(blobs[0], blobs[1])

    def test_reject_foreign_and_symlink_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'root'
            root.mkdir()
            foreign = Path(temporary) / 'foreign.py'
            foreign.write_text('VALUE = 7\n')
            with self.assertRaises(ValueError):
                B.compile_sources(root, [foreign])
            link = root / 'link.py'
            link.symlink_to(foreign)
            with self.assertRaises(ValueError):
                B.compile_sources(root, [link])

    def test_sealed_runtime_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / 'runtime.json'
            marker.write_text('{"frozen":true}\n')
            before = marker.read_bytes()
            result = subprocess.run([sys.executable, '-I', '-B', str(Path(B.__file__)), str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refuse to modify a sealed', result.stderr)
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual(list(root.iterdir()), [marker])


if __name__ == '__main__':
    unittest.main()
