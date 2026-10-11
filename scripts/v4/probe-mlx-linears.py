#!/usr/bin/env python3
"""Isolate packed MLX linears inside unchanged MPS aggregator arithmetic.

Research bridge uses explicit host copies between GPU frameworks, no CPU model
calculation. This is not an application backend or a performance claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Runtime'))
import mlx.core as mx
import numpy as np
from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--heads', choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--mode', choices=['packed', 'dequantized-reference'], default='packed',
                        help='Dequantized reference only isolates weight error; never accepted as low-bit execution')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'inputs').symlink_to(args.dataset.resolve(), target_is_directory=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    calls = {}

    class PackedLinear(torch.nn.Module):
        def __init__(self, name, original, checkpoint, recipe):
            super().__init__()
            self.name = name
            self.weight_parameters = recipe['inFeatures'] * recipe['outFeatures']
            self.q, self.scales, self.offsets = [mx.array(checkpoint.get_tensor(name + suffix))
                                               for suffix in ['.qweight', '.scales', '.offsets']]
            assert self.q.dtype == mx.uint32 and self.scales.dtype == mx.float16 and self.offsets.dtype == mx.float16
            assert recipe['bits'] == 8 and recipe['groupSize'] == 64
            self.bias = torch.nn.Parameter(torch.empty_like(original.bias), requires_grad=False) if original.bias is not None else None
            mx.eval(self.q, self.scales, self.offsets)
            if args.mode == 'dequantized-reference':
                weight = mx.dequantize(self.q, self.scales, self.offsets, bits=8, group_size=64, mode='affine')
                mx.eval(weight)
                self.floating_weight = torch.from_numpy(np.asarray(weight).copy()).to('mps')
                self.q = self.scales = self.offsets = None

        def forward(self, x):
            if args.mode == 'dequantized-reference':
                return torch.nn.functional.linear(x, self.floating_weight, self.bias)
            source = mx.array(x.detach().cpu().numpy())
            result = mx.quantized_matmul(source, self.q, self.scales, self.offsets, bits=8, group_size=64, mode='affine')
            mx.eval(result)
            output = torch.from_numpy(np.asarray(result).copy()).to('mps')
            record = calls.setdefault(self.name, {'calls': 0, 'originalWeightParameters': self.weight_parameters})
            record['calls'] += 1
            return output + self.bias if self.bias is not None else output

    class Aggregator:
        def __init__(self, weights, meta, bits=8, group_size=64, weight_info=None):
            assert bits == 8 and group_size == 64 and weight_info['format'] == 'rebuild3d-vggt-mlx'
            expected = {f'aggregator.{name}' for name, module in meta.named_modules() if isinstance(module, torch.nn.Linear)}
            assert expected == set(weight_info['quantizedModules'])
            with safe_open(weights, framework='np') as checkpoint:
                for name, recipe in weight_info['quantizedModules'].items():
                    local = name.removeprefix('aggregator.')
                    original = meta.get_submodule(local)
                    packed = PackedLinear(name, original, checkpoint, recipe)
                    owner, leaf = local.rsplit('.', 1)
                    setattr(meta.get_submodule(owner), leaf, packed)
            with safe_open(weights, framework='pt', device='cpu') as checkpoint:
                for name, parameter in meta.named_parameters():
                    value = checkpoint.get_tensor('aggregator.' + name)
                    assert value.shape == parameter.shape and value.dtype == torch.float16
                    if '.' in name:
                        owner, leaf = name.rsplit('.', 1)
                        target = meta.get_submodule(owner)
                    else:
                        target, leaf = meta, name
                    setattr(target, leaf, torch.nn.Parameter(value.to('mps'), requires_grad=False))
            for name, value in [('_resnet_mean', [.485, .456, .406]), ('_resnet_std', [.229, .224, .225])]:
                setattr(meta, name, torch.tensor(value, dtype=torch.float16, device='mps').view(1, 1, 3, 1, 1))
            assert not any(p.is_meta for p in list(meta.parameters()) + list(meta.buffers()))
            self.model = meta.eval()

        def __call__(self, images, progress=None):
            source = torch.from_numpy(np.asarray(images).copy()).to('mps')
            with torch.inference_mode():
                tokens, patch_start = self.model(source)
            result = [mx.array(token.cpu().numpy()) if token is not None else None for token in tokens]
            mx.eval(result)
            if progress:
                progress('MPS-arithmetic-MLX-packed-linears-complete')
            return result, patch_start

        def audit(self):
            return {'backend': 'experimental-mps-arithmetic-mlx-packed-linears', 'scope': __doc__,
                    'mode': args.mode,
                    'linears': calls, 'packedKernelCalls': sum(c['calls'] for c in calls.values()),
                    'executedLowBitParameters': sum(c['originalWeightParameters'] for c in calls.values() if c['calls']),
                    'prepacked': True}

    spec = importlib.util.spec_from_file_location('bridge_experiment', Path(__file__).with_name('infer-mlx-experiment.py'))
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    experiment.Aggregator = Aggregator
    report = {'status': 'running', 'scope': __doc__, 'mode': args.mode, 'heads': args.heads,
              'scriptSHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        sys.argv = ['linear-bridge-probe', str(args.dataset), str(args.output / 'inference'), '--repo', str(args.repo),
                    '--weights', str(args.weights), '--bits', '8', '--heads', args.heads]
        experiment.main()
        report['status'] = 'linear-bridge-inference-completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['packedKernelCalls'] = sum(c['calls'] for c in calls.values())
        report['executedLowBitParameters'] = sum(c['originalWeightParameters'] for c in calls.values() if c['calls'])
        report['executedLowBitFraction'] = report['executedLowBitParameters'] / 1157941492
        (args.output / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
