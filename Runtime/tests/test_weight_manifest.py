import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from rebuild3d_worker.model_weights import METADATA_KEY, digest, load_model, make_recipe, verify_weights


class ManifestTests(unittest.TestCase):
    def toy(self):
        model = torch.nn.Module()
        model.aggregator = torch.nn.Sequential(torch.nn.Linear(128,64), torch.nn.LayerNorm(64))
        model.aggregator.register_parameter('not_a_linear', torch.nn.Parameter(torch.zeros(64,128)))
        model.camera_head = torch.nn.Linear(128,64)
        return model

    def test_only_actual_aggregator_linear_weights_are_quantized(self):
        recipe = make_recipe(self.toy(), 'w8')
        self.assertEqual(set(recipe['quantizedModules']), {'aggregator.0'})
        self.assertEqual(recipe['modulePrecisionMap']['aggregator.not_a_linear'], 'FP16')
        self.assertEqual(recipe['modulePrecisionMap']['camera_head.weight'], 'FP32')

    def test_corrupt_weights_and_disagreeing_recipes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'vggt.safetensors'
            recipe = make_recipe(self.toy(), 'trimmed-fp16')
            save_file({'tiny':torch.zeros(32)},path,metadata={METADATA_KEY:json.dumps(recipe)})
            manifest = {'recipe':recipe, 'weightBytes':path.stat().st_size,'derivedWeightSHA256':digest(path)}
            path.with_suffix('.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_weights(path)['policy'], 'trimmed-fp16')
            manifest['recipe']['policy'] = 'w8'
            path.with_suffix('.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'recipe disagree'):
                verify_weights(path)
            with path.open('r+b') as stream:
                stream.seek(-1,2); stream.write(b'\x01')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify_weights(path)

    def test_derived_checkpoint_never_silently_falls_back_to_cpu(self):
        with self.assertRaisesRegex(ValueError, 'MPS-only'):
            load_model(Path('unused'),Path('unused'),'cpu',info={'format':'rebuild3d-vggt'})

    def test_mlx_packing_never_enters_the_mps_packed_loader(self):
        with self.assertRaisesRegex(ValueError, 'MLXBackend loader'):
            load_model(Path('unused'),Path('unused'),'mps',info={'format':'rebuild3d-vggt-mlx'})


if __name__ == '__main__':
    unittest.main()
