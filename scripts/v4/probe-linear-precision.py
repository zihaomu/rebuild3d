#!/usr/bin/env python3
"""Research packed W8 factor precision and rounding; never a production loader.

Both policies execute MLX packed uint32 matrix multiplication with FP32 inputs,
factors and output, add the bias in FP32, then round to the incoming FP16 dtype.
The promoted-factors policy preserves the offline FP16 packing exactly. The
recomputed-factors policy packs the verified FP16 source weights using FP32
arithmetic during development loading. It is not a deployable packed artifact.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import mlx.core as mx
import mlx_aggregator
from mlx_block import Block


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--group-size', type=int, choices=[32, 64], default=32)
    parser.add_argument('--heads', choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--policy', choices=['promoted-factors', 'recomputed-factors'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'inputs').symlink_to(args.dataset.resolve(), target_is_directory=True)
    source_paths = [Path(__file__), Path(__file__).with_name('infer-mlx-experiment.py'),
                    Path(__file__).with_name('mlx_aggregator.py'), Path(__file__).with_name('mlx_block.py')]
    for path in source_paths:
        (args.output / path.name).write_bytes(path.read_bytes())
    calls = {}
    storage = []

    class PrecisionBlock(Block):
        def __init__(self, weights, config, bits=0, group_size=64):
            assert bits == 8 and group_size == args.group_size
            if args.policy == 'promoted-factors':
                assert all(name + '.qweight' in weights for name in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'))
                super().__init__(weights, config, bits, group_size)
                self.packed = {name: (q, s.astype(mx.float32), b.astype(mx.float32))
                               for name, (q, s, b) in self.packed.items()}
            else:
                assert not any(name.endswith('.qweight') for name in weights)
                super().__init__(weights, config, 0, group_size)
                self.bits = bits
                for name in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'):
                    source = self.weights.pop(name + '.weight')
                    assert source.dtype == mx.float16
                    self.packed[name] = mx.quantize(source.astype(mx.float32), bits=bits,
                                                    group_size=group_size, mode='affine')
            mx.eval(self.packed)
            self.block_index = len(storage)
            storage.append(self.storage())

        def linear(self, name, x):
            if name not in self.packed:
                return super().linear(name, x)
            q, scales, offsets = self.packed[name]
            assert x.dtype == mx.float16 and q.dtype == mx.uint32
            assert scales.dtype == offsets.dtype == mx.float32
            value = mx.quantized_matmul(x.astype(mx.float32), q, scales, offsets,
                                        bits=self.bits, group_size=self.group_size, mode='affine')
            assert value.dtype == mx.float32
            if name + '.bias' in self.weights:
                value = value + self.weights[name + '.bias'].astype(mx.float32)
            key = f'{self.block_index}.{name}'
            record = calls.setdefault(key, {'calls': 0, 'originalWeightParameters': q.shape[0] * q.shape[1] * 32 // self.bits})
            record['calls'] += 1
            return value.astype(x.dtype)

    mlx_aggregator.Block = PrecisionBlock
    spec = importlib.util.spec_from_file_location('precision_experiment', Path(__file__).with_name('infer-mlx-experiment.py'))
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    report = {'status': 'running', 'scope': __doc__, 'policy': args.policy,
              'groupSize': args.group_size, 'heads': args.heads,
              'sourcesSHA256': {p.name: digest(p) for p in source_paths}}
    try:
        sys.argv = ['linear-precision-probe', str(args.dataset), str(args.output / 'inference'),
                    '--repo', str(args.repo), '--weights', str(args.weights), '--bits', '8',
                    '--group-size', str(args.group_size), '--heads', args.heads]
        experiment.main()
        report['status'] = 'linear-precision-inference-completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['packedKernelCalls'] = sum(c['calls'] for c in calls.values())
        report['executedLowBitParameters'] = sum(c['originalWeightParameters'] for c in calls.values() if c['calls'])
        report['executedLowBitFraction'] = report['executedLowBitParameters'] / 1157941492
        report['linears'] = calls
        report['blocks'] = storage
        report['packedBytesIncludingFP32Factors'] = sum(b['packedBytesIncludingScales'] for b in storage)
        (args.output / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
