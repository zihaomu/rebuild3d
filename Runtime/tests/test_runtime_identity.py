"""Exercise production checkpoint decisions with small files and stub stage work.

This covers invalidation, not GPU inference, recovery UI, or a finished model.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rebuild3d_worker import worker as W


STAGES = {'prepare': 'inputs', 'inference': 'inference', 'fusion': 'fusion',
          'unwrap': 'uv-native', 'atlas': 'uv', 'texture': 'texture', 'package': 'bundle'}


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.component = self.root / 'component'
        self.component.mkdir()
        self.paths = {}
        for name in ('worker.py', 'uv_layout.dylib', 'native', 'raster', 'weights', 'weights.json', 'repo/vggt/model.py'):
            path = self.component / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}' if name.endswith('.json') else 'fixture:' + name)
            self.paths[name] = path
        photos = []
        for i in range(3):
            path = self.root / f'photo-{i}.bin'
            path.write_bytes(bytes([i]))
            photos.append({'id': str(i), 'name': path.name, 'sourcePath': str(path), 'sourceSHA256': W.sha(path)})
        self.manifest = {'formatVersion': 1, 'jobID': 'identity-test', 'photos': photos,
                         'runtime': {'nativeWorker': str(self.paths['native']), 'rasterLibrary': str(self.paths['raster']),
                                     'weights': str(self.paths['weights']), 'vggtRepository': str(self.component / 'repo')},
                         'parameters': {'modelSize': 518, 'textureSize': 4096, 'volumeResolution': 256}}
        self.job = self.root / 'job'

    def run_fixture(self):
        executed = []

        def execute(supervisor, stage, command, environment, memory_limit):
            executed.append(stage)
            output = supervisor.root / STAGES[stage]
            output.mkdir()
            (output / 'artifact.bin').write_bytes(b'fixture stage output')
            return {'stageElapsedSeconds': 0, 'sampledProcessTreePeakRSSBytes': 0}

        # Only the hash fixture location and heavy stage work are substituted.
        # Identity construction, signatures, checkpoint hashing, validation,
        # downstream invalidation and artifact preservation run production code.
        with patch.object(W, '__file__', str(self.paths['worker.py'])), \
                patch.object(W.Supervisor, 'execute', execute), contextlib.redirect_stdout(io.StringIO()):
            W.Supervisor(self.manifest, self.job).run()
        return executed, json.loads((self.job / 'runtime-identity.json').read_text())['identity']

    def test_each_runtime_dependency_change_invalidates_completed_stages(self):
        stages, identity = self.run_fixture()
        self.assertEqual(stages, list(STAGES))
        self.assertEqual(self.run_fixture(), ([], identity))
        changes = [('weights', 'different weight bytes'),
                   ('weights.json', json.dumps({'recipe': {'format': 'rebuild3d-vggt-mlx', 'backendID': 'mlx-aggregator-pytorch-heads-v1', 'groupSize': 32}})),
                   ('weights.json', json.dumps({'recipe': {'format': 'rebuild3d-vggt-mlx', 'backendID': 'mlx-aggregator-pytorch-heads-v1', 'groupSize': 64, 'linearComputePolicy': 'changed'}})),
                   ('worker.py', 'changed worker implementation'), ('repo/vggt/model.py', 'changed model implementation'),
                   ('uv_layout.dylib', 'changed native UV implementation'), ('native', 'changed native worker'),
                   ('raster', 'changed raster implementation')]
        for filename, content in changes:
            with self.subTest(dependency=filename, content=content):
                self.paths[filename].write_text(content)
                stages, new_identity = self.run_fixture()
                self.assertNotEqual(new_identity, identity)
                self.assertEqual(stages, list(STAGES))
                self.assertEqual(self.run_fixture(), ([], new_identity))
                identity = new_identity

    def test_corrupt_inference_preserves_prepare_but_rebuilds_downstream(self):
        _, identity = self.run_fixture()
        before = W.sha(self.job / 'inputs/artifact.bin')
        (self.job / 'inference/artifact.bin').write_bytes(b'corrupt')
        stages, new_identity = self.run_fixture()
        self.assertEqual(new_identity, identity)
        self.assertEqual(stages, list(STAGES)[1:])
        self.assertEqual(W.sha(self.job / 'inputs/artifact.bin'), before)
        self.assertEqual(self.run_fixture(), ([], identity))
        preserved = list((self.job / 'attempts').glob('inference-*/artifact.bin'))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b'corrupt')


if __name__ == '__main__':
    unittest.main()
