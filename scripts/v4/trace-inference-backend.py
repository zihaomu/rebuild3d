#!/usr/bin/env python3
"""Record bounded intermediate samples while running unchanged inference math.

Diagnostic only: observations synchronize devices and invalidate timing claims.
No source, checkpoint or frozen application is modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Runtime'))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--backend', choices=['torch', 'mlx-float', 'mlx-w8'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'inputs').symlink_to(args.dataset.resolve(), target_is_directory=True)
    samples = args.output / 'trace'
    samples.mkdir()
    count = len(json.loads((args.dataset / 'dataset.json').read_text())['records'])
    report = {'status': 'running', 'scope': __doc__, 'backend': args.backend,
              'startedAtUTC': datetime.now(timezone.utc).isoformat(), 'scriptSHA256': sha(__file__),
              'datasetSHA256': sha(args.dataset / 'dataset.json'), 'weightSHA256': sha(args.weights),
              'records': []}
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())

    def save():
        (args.output / 'trace-report.json').write_text(json.dumps(report, indent=2) + '\n')

    def record(name, value):
        shape = list(value.shape)
        length = shape[-2]
        views = count if '.global_blocks.' in name else 1
        if length % views:
            raise ValueError('Unexpected global sequence layout')
        per_view = length // views
        local = np.unique(np.r_[np.arange(min(5, per_view)),
                                np.linspace(min(5, per_view - 1), per_view - 1, min(64, per_view), dtype=int)])
        indices = np.concatenate([local + view * per_view for view in range(views)])
        if isinstance(value, torch.Tensor):
            subset = value.index_select(-2, torch.tensor(indices, device=value.device)).detach().cpu().numpy()
        else:
            import mlx.core as mx
            subset = np.asarray(mx.take(value, mx.array(indices.astype(np.int32)), axis=-2))
        path = samples / (name + '.npz')
        np.savez_compressed(path, values=subset, indices=indices, originalShape=shape)
        report['records'].append({'name': name, 'shape': shape, 'sampleShape': list(subset.shape),
                                  'sha256': sha(path), 'dtype': str(subset.dtype)})
        save()

    def camera_hook(module, inputs, output):
        tokens = inputs[0][-1]
        np.savez_compressed(samples / 'camera.npz',
                            tokens=tokens[:, :, :1].detach().cpu().numpy(),
                            iterations=np.stack([p.detach().cpu().numpy() for p in output]))

    try:
        from rebuild3d_worker import infer
        if args.backend == 'torch':
            original_load = infer.load_model

            def traced_load(*a, **kw):
                model, ignored = original_load(*a, **kw)
                for group in ['patch_embed.blocks', 'frame_blocks', 'global_blocks']:
                    for index, block in enumerate(model.aggregator.get_submodule(group)):
                        name = 'aggregator.' + group + '.' + str(index)
                        if group == 'patch_embed.blocks' and index == 0:
                            block.register_forward_pre_hook(lambda m, inputs: record('dino-input', inputs[0]))
                        block.register_forward_hook(lambda m, inputs, output, n=name: record(n, output))
                model.camera_head.register_forward_hook(camera_hook)
                return model, ignored

            infer.load_model = traced_load
            sys.argv = ['traced-production-inference', str(args.dataset), str(args.output / 'inference'),
                        '--repo', str(args.repo), '--weights', str(args.weights), '--size', '518',
                        '--count', str(count), '--max-memory-gib', '10.5']
            infer.main()
        else:
            spec = importlib.util.spec_from_file_location('v4_experiment', Path(__file__).with_name('infer-mlx-experiment.py'))
            experiment = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(experiment)
            from mlx_block import Block
            original_call = Block.__call__
            original_aggregator = experiment.Aggregator
            original_model = experiment.meta_model
            names = {}

            def traced_call(block, x, positions=None):
                name = names[id(block)]
                if name == 'aggregator.patch_embed.blocks.0':
                    record('dino-input', x)
                result = original_call(block, x, positions)
                record(name, result)
                return result

            class TracedAggregator(original_aggregator):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    for group, blocks in self.blocks.items():
                        for index, block in enumerate(blocks):
                            names[id(block)] = 'aggregator.' + group + '.' + str(index)

            def traced_model(*a, **kw):
                model = original_model(*a, **kw)
                model.camera_head.register_forward_hook(camera_hook)
                return model

            Block.__call__ = traced_call
            experiment.Aggregator = TracedAggregator
            experiment.meta_model = traced_model
            sys.argv = ['traced-experimental-inference', str(args.dataset), str(args.output / 'inference'),
                        '--repo', str(args.repo), '--weights', str(args.weights),
                        '--bits', '0' if args.backend == 'mlx-float' else '8', '--heads', 'fp32']
            experiment.main()
        report['cameraTraceSHA256'] = sha(samples / 'camera.npz')
        report['status'] = 'traced-inference-completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finishedAtUTC'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
