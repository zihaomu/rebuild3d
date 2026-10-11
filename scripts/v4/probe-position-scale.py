#!/usr/bin/env python3
"""Isolated hypothesis: normalize low-resolution RoPE patch centers to 518.

No production file is changed. Uses the real quantized GPU backend and writes
an explicit experiment record; this is not an accepted model configuration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
import torch
from rebuild3d_worker import infer, mlx_block


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--size', type=int, default=392)
    parser.add_argument('--reference-size', type=int, default=518)
    parser.add_argument('--backend', choices=('mlx', 'floating'), default='mlx')
    parser.add_argument('--count', type=int, default=7)
    args = parser.parse_args()
    assert args.size == 392 and args.reference_size == 518
    assert 3 <= args.count <= 12
    assert not args.output.exists()
    ratio = args.reference_size / args.size
    experiment = {'status': 'running', 'startedAtUTC': datetime.now(timezone.utc).isoformat(),
                  'scope': 'Experimental RoPE coordinates only; not production or performance acceptance',
                  'modelSize': args.size, 'referenceSize': args.reference_size, 'backend': args.backend,
                  'photoCount': args.count,
                  'formula': 'where(position > 0, (position - 0.5) * (518/392) + 0.5, 0)',
                  'scriptSHA256': sha(__file__), 'originalBlockSHA256': sha(mlx_block.__file__),
                  'inferSHA256': sha(infer.__file__), 'ropeCalls': 0}

    if args.backend == 'mlx':
        owner, attribute = mlx_block.Block, 'rope'
        original = owner.rope

        def scaled(self, tokens, positions):
            experiment['ropeCalls'] += 1
            positions = positions.astype(mx.float32)
            adjusted = mx.where(positions > 0, (positions - .5) * ratio + .5, 0)
            return original(self, tokens, adjusted)
    else:
        sys.path.insert(0, str(args.repo.resolve()))
        from vggt.layers import rope
        owner, attribute = rope.RotaryPositionEmbedding2D, '_compute_frequency_components'
        original = owner._compute_frequency_components
        experiment['originalTorchRopeSHA256'] = sha(rope.__file__)

        def scaled(self, dim, seq_len, device, dtype):
            # Keep pinned frequency construction, FP16 angle rounding and table
            # lookup; only its position centers change. No model CPU fallback.
            experiment['ropeCalls'] += 1
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            positions = torch.where(positions > 0, (positions - .5) * ratio + .5, 0)
            angles = torch.einsum('i,j->ij', positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            return angles.cos().to(dtype), angles.sin().to(dtype)

    setattr(owner, attribute, scaled)
    sys.argv = ['experimental-infer', str(args.dataset), str(args.output),
                '--repo', str(args.repo), '--weights', str(args.weights),
                '--size', str(args.size), '--count', str(args.count), '--max-memory-gib', '10.5']
    try:
        infer.main()
        assert experiment['ropeCalls'] == 96
        config = json.loads((args.output / 'config.json').read_text())
        expected = 'pytorch-mps-v3' if args.backend == 'floating' else 'mlx-aggregator-pytorch-heads-v1'
        assert config['backendID'] == expected
        experiment['status'] = 'inference-complete-requires-geometry-review'
    except BaseException as error:
        experiment.update(status='failed', error=repr(error))
        raise
    finally:
        setattr(owner, attribute, original)
        experiment['finishedAtUTC'] = datetime.now(timezone.utc).isoformat()
        args.output.mkdir(exist_ok=True)
        (args.output / 'position-experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')


if __name__ == '__main__':
    main()
