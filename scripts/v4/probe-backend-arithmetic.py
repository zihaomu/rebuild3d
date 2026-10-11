#!/usr/bin/env python3
"""Isolate MPS/MLX preprocessing and rotary arithmetic; never a shipped backend.

Reuses verified weights and identical photos. Explicit GPU transfers are included
in the experiment. Results cannot be called application or speed acceptance.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Runtime'))
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--bits', type=int, choices=[0, 8], default=0)
    parser.add_argument('--group-size', type=int, choices=[32, 64, 128], default=64)
    parser.add_argument('--heads', choices=['fp16', 'fp32'], default='fp32')
    parser.add_argument('--dino-prefix', type=int, choices=[0, 8, 16], default=0,
                        help='Diagnostic MPS floating DINO prefix; remaining blocks retain requested precision')
    parser.add_argument('--floating-weights', type=Path,
                        help='Verified trimmed checkpoint for the floating prefix; research dependency only')
    parser.add_argument('--mode', choices=['normalization', 'frontend', 'rope', 'frontend-rope'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'inputs').symlink_to(args.dataset.resolve(), target_is_directory=True)
    spec = importlib.util.spec_from_file_location('precision_experiment', Path(__file__).with_name('infer-mlx-experiment.py'))
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    original = experiment.Aggregator
    source = textwrap.dedent(inspect.getsource(original.__call__))
    old_normalization = "images=(images-mx.array([.485,.456,.406],dtype=images.dtype)[None,None,:,None,None])\n    images=images/mx.array([.229,.224,.225],dtype=images.dtype)[None,None,:,None,None]"
    old_convolution = "tokens=mx.conv2d(image,weight,stride=14)+self.weights['patch_embed.patch_embed.proj.bias']"
    calls = {'normalization': 0, 'convolution': 0, 'rotaryTable': 0, 'floatingPrefixBlocks': 0,
             'quantizedMatmul': 0, 'executedLowBitParameters': 0}
    if args.dino_prefix and (args.floating_weights is None or args.mode not in ['frontend', 'frontend-rope']):
        raise ValueError('Floating prefix requires explicit checkpoint and matching MPS frontend')
    if args.dino_prefix:
        from rebuild3d_worker.model_weights import verify_weights
        floating_info = verify_weights(args.floating_weights)
        if floating_info.get('policy') != 'trimmed-fp16':
            raise ValueError('Prefix requires verified trimmed FP16 parameters')

    def tensor(value):
        return torch.from_numpy(np.asarray(value).copy()).to('mps')

    def normalize(images):
        with torch.inference_mode():
            x = tensor(images)
            mean = torch.tensor([.485, .456, .406], dtype=x.dtype, device='mps').view(1, 1, 3, 1, 1)
            std = torch.tensor([.229, .224, .225], dtype=x.dtype, device='mps').view(1, 1, 3, 1, 1)
            result = mx.array(((x - mean) / std).cpu().numpy())
            mx.eval(result)
        calls['normalization'] += 1
        return result

    def convolution(image, weight, bias):
        with torch.inference_mode():
            result = F.conv2d(tensor(image).permute(0, 3, 1, 2), tensor(weight).permute(0, 3, 1, 2),
                              tensor(bias), stride=14)
            result = mx.array(result.permute(0, 2, 3, 1).cpu().numpy())
            mx.eval(result)
        calls['convolution'] += 1
        return result

    if args.mode in ['normalization', 'frontend', 'frontend-rope']:
        assert source.count(old_normalization) == 1
        source = source.replace(old_normalization, 'images=normalize(images)')
    if args.mode in ['frontend', 'frontend-rope']:
        assert source.count(old_convolution) == 1
        source = source.replace(old_convolution, "tokens=convolution(image,weight,self.weights['patch_embed.patch_embed.proj.bias'])")
    if args.dino_prefix:
        old_loop = "for i,block in enumerate(self.blocks['patch_embed.blocks']):\n        tokens=block(tokens);mx.eval(tokens)"
        new_loop = "tokens=self.mps_prefix(tokens)\n    for i,block in enumerate(self.blocks['patch_embed.blocks']):\n        if i < self.prefix_count: continue\n        tokens=block(tokens);mx.eval(tokens)"
        assert source.count(old_loop) == 1
        source = source.replace(old_loop, new_loop)
    namespace = {'mx': mx, 'normalize': normalize, 'convolution': convolution}
    exec(compile(source, str(args.output / 'aggregator-call.py'), 'exec'), namespace)

    class ProbeAggregator(original):
        __call__ = namespace['__call__']

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.prefix_count = args.dino_prefix
            self.prefix = []
            if self.prefix_count:
                from safetensors import safe_open
                meta = a[1]
                with safe_open(args.floating_weights, framework='pt', device='cpu') as checkpoint:
                    for i in range(self.prefix_count):
                        block = meta.patch_embed.blocks[i]
                        for name, parameter in block.named_parameters():
                            value = checkpoint.get_tensor(f'aggregator.patch_embed.blocks.{i}.{name}')
                            assert value.dtype == torch.float16 and value.shape == parameter.shape
                            owner, leaf = name.rsplit('.', 1)
                            setattr(block.get_submodule(owner), leaf, torch.nn.Parameter(value.to('mps'), requires_grad=False))
                        block.eval()
                        self.prefix.append(block)
                        size = sum(p.numel() * p.element_size() for p in block.parameters())
                        class PrefixAudit:
                            def storage(self, byte_count=size):
                                return {'bits': 0, 'operation': 'pytorch-mps-fp16-dino-prefix',
                                        'floatBytes': byte_count, 'packedLinearCount': 0, 'packedBytesIncludingScales': 0}
                        self.blocks['patch_embed.blocks'][i] = PrefixAudit()
                mx.clear_cache()

        def mps_prefix(self, tokens):
            with torch.inference_mode():
                x = tensor(tokens)
                for block in self.prefix:
                    x = block(x)
                    calls['floatingPrefixBlocks'] += 1
                result = mx.array(x.cpu().numpy())
                mx.eval(result)
                return result

    experiment.Aggregator = ProbeAggregator
    from mlx_block import Block
    original_linear = Block.linear
    def counted_linear(block, name, x):
        if name in block.packed:
            q, _, _ = block.packed[name]
            calls['quantizedMatmul'] += 1
            calls['executedLowBitParameters'] += q.shape[0] * q.shape[1] * 32 // block.bits
        return original_linear(block, name, x)
    Block.linear = counted_linear
    if args.mode in ['rope', 'frontend-rope']:
        cache = {}

        def rope(block, tokens, positions):
            dim = tokens.shape[-1] // 2
            key = (dim, block.config['ropeFrequency'], str(tokens.dtype))
            if key not in cache:
                # Same FP32 frequency and FP16 trig sequence as pinned Torch RoPE.
                with torch.inference_mode():
                    exponent = torch.arange(0, dim, 2, device='mps').float() / dim
                    inverse = 1.0 / (block.config['ropeFrequency'] ** exponent)
                    angles = torch.einsum('i,j->ij', torch.arange(38, device='mps', dtype=inverse.dtype), inverse)
                    angles = angles.to(torch.float16)
                    angles = torch.cat((angles, angles), dim=-1)
                    cache[key] = (mx.array(angles.cos().cpu().numpy()), mx.array(angles.sin().cpu().numpy()))
                    mx.eval(cache[key])
                calls['rotaryTable'] += 1
            cosine, sine = cache[key]
            results = []
            for axis in [0, 1]:
                index = positions[..., axis]
                cos = cosine[index][:, None, :, :]
                sin = sine[index][:, None, :, :]
                part = tokens[..., axis * dim:(axis + 1) * dim]
                rotated = mx.concatenate((-part[..., dim // 2:], part[..., :dim // 2]), axis=-1)
                results.append(part * cos + rotated * sin)
            return mx.concatenate(results, axis=-1)

        Block.rope = rope
    (args.output / 'aggregator-call.py').write_text(source)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    report = {'status': 'running', 'mode': args.mode, 'bits': args.bits, 'heads': args.heads, 'groupSize': args.group_size,
              'floatingDinoPrefix': args.dino_prefix, 'scope': __doc__,
              'scriptSHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'originalAggregatorSHA256': hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest(),
              'generatedCallSHA256': hashlib.sha256(source.encode()).hexdigest()}
    try:
        sys.argv = ['arithmetic-probe', str(args.dataset), str(args.output / 'inference'), '--repo', str(args.repo),
                    '--weights', str(args.weights), '--bits', str(args.bits), '--group-size', str(args.group_size), '--heads', args.heads]
        experiment.main()
        report['status'] = 'arithmetic-probe-completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['calls'] = calls
        report['executedLowBitParameterFraction'] = calls['executedLowBitParameters'] / 1157941492
        if args.floating_weights:
            with args.floating_weights.open('rb') as stream:
                report['researchFloatingWeightSHA256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            report['dependencyScope'] = 'Two research checkpoints required by this probe; not an independently packaged hybrid weight artifact.'
        (args.output / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
