"""Prepacked MLX VGGT aggregator for the managed sparse reconstruction backend.

Matches pinned Meta VGGT aggregator/DINO ordering, including separate first-view
tokens and the four cached layers. Output heads use their manifest-declared precision in PyTorch/MPS.
"""
import mlx.core as mx
from safetensors import safe_open

import numpy as np
import torch

from .mlx_block import Block, FP16_LINEAR_POLICY


ROTARY_POSITION_POLICY = "reference-518-patch-centers-v1"
EXECUTION_POLICY = "async-block-eval-final-sync-v1"


def rotary_positions(count, size):
    """Use the same image-relative patch centers at both supported resolutions."""
    if size not in (518, 392):
        raise ValueError('Managed MLX supports 518 and 392 pixel inputs')
    patch_side = size // 14
    patch_count = patch_side ** 2
    grid = mx.stack(mx.meshgrid(mx.arange(patch_side), mx.arange(patch_side),
                               indexing='ij'), axis=-1).reshape(1, patch_count, 2) + 1
    positions = mx.concatenate((mx.zeros((count, 5, 2), dtype=mx.int32),
                                mx.broadcast_to(grid, (count, patch_count, 2))), axis=1)
    if size != 518:
        positions = positions.astype(mx.float32)
        positions = mx.where(positions > 0, (positions - .5) * (518 / size) + .5, 0)
    return positions


class Aggregator:
    def __init__(self, weights, meta, bits=8, group_size=None, weight_info=None, size=518):
        self.blocks={};self.weights={};self.bits=bits
        self.block_configs={};self.calls=0
        self.prepacked=weight_info is not None and weight_info['format']=='rebuild3d-vggt-mlx'
        if not self.prepacked or bits != 8:
            raise ValueError('Managed MLX requires prepacked W8 weights')
        if size not in (518,392):
            raise ValueError('Managed MLX supports 518 and 392 pixel inputs')
        groups = {recipe['groupSize'] for recipe in weight_info['quantizedModules'].values()}
        if len(groups) != 1 or (group_size is not None and groups != {group_size}):
            raise ValueError('MLX requires one consistent declared quantization group size')
        self.group_size = group_size = next(iter(groups))
        self.linear_policy = weight_info.get('linearComputePolicy', FP16_LINEAR_POLICY)
        self.size = size
        with safe_open(weights,framework='np') as checkpoint:
            available={k[11:] for k in checkpoint.keys() if k.startswith('aggregator.')}
            expected={k for k,_ in meta.named_parameters()}
            if self.prepacked:
                modules=weight_info['quantizedModules']
                actual={name for name,mod in meta.named_modules() if isinstance(mod,torch.nn.Linear)}
                if actual!={name.removeprefix('aggregator.') for name in modules}:
                    raise ValueError('Prepacked module list differs from all aggregator Linears')
                for name,recipe in modules.items():
                    local=name.removeprefix('aggregator.');module=meta.get_submodule(local)
                    if recipe['bits']!=bits or recipe['groupSize']!=group_size or recipe['packing']!='mlx-affine-uint32-v1' or recipe['inFeatures']!=module.in_features or recipe['outFeatures']!=module.out_features:
                        raise ValueError('MLX packing policy/layout mismatch')
                    expected.remove(local+'.weight')
                    expected.update(local+suffix for suffix in ('.qweight','.scales','.offsets'))
            if available!=expected:
                raise ValueError('Aggregator tensor set differs from pinned architecture')
            # Validate every tensor against the pinned architecture before GPU loading.
            parameters = dict(meta.named_parameters())
            for name in expected:
                tensor = checkpoint.get_tensor('aggregator.' + name)
                if name in parameters:
                    shape, dtype = tuple(parameters[name].shape), np.dtype('float16')
                else:
                    local, suffix = name.rsplit('.', 1)
                    recipe = modules['aggregator.' + local]
                    shape = (recipe['outFeatures'], recipe['inFeatures'] //
                             (32 // bits if suffix == 'qweight' else group_size))
                    dtype = np.dtype('uint32' if suffix == 'qweight' else 'float16')
                if tensor.shape != shape or tensor.dtype != dtype:
                    raise ValueError('MLX tensor shape/precision mismatch: ' + name)
                del tensor
            for group in ('patch_embed.blocks','frame_blocks','global_blocks'):
                group_blocks=[]
                for index,source in enumerate(meta.get_submodule(group)):
                    prefix=f'{group}.{index}.'
                    names=[k for k in available if k.startswith(prefix)]
                    tensors={k[len(prefix):]:mx.array(checkpoint.get_tensor('aggregator.'+k)) for k in names}
                    config={'heads':source.attn.num_heads,'norm1Epsilon':source.norm1.eps,'norm2Epsilon':source.norm2.eps,
                            'qkNormEpsilon':getattr(source.attn.q_norm,'eps',None),
                            'ropeFrequency':getattr(source.attn.rope,'base_frequency',None),
                            'activation':type(source.mlp.act).__name__,'activationApproximation':getattr(source.mlp.act,'approximate',None)}
                    group_blocks.append(Block(tensors,config,bits,group_size,self.linear_policy))
                    self.block_configs[prefix[:-1]]=config
                    available.difference_update(names)
                    del tensors
                self.blocks[group]=group_blocks
            for name in available:
                self.weights[name]=mx.array(checkpoint.get_tensor('aggregator.'+name))
            mx.eval(self.weights)
        if size != 518:
            # Static learned-position preprocessing, not a model/quantized CPU fallback.
            # Reuse the pinned DINO interpolation including offset and antialias policy.
            position = torch.from_numpy(np.asarray(self.weights['patch_embed.pos_embed']).copy())
            meta.patch_embed.pos_embed = torch.nn.Parameter(position, requires_grad=False)
            dummy = torch.empty((1, (size//14)**2 + 1, 1024), dtype=torch.float16)
            with torch.inference_mode():
                position = meta.patch_embed.interpolate_pos_encoding(dummy, size, size)
            self.weights['patch_embed.pos_embed'] = mx.array(position.numpy())
            mx.eval(self.weights['patch_embed.pos_embed'])
        self.patch_norm_eps=meta.patch_embed.norm.eps
        self.cached_layers=set(meta.cached_layer_indices)
        self.depth=meta.depth

    def __call__(self,images,progress=None):
        if images.ndim!=5 or images.shape[0]!=1 or tuple(images.shape[2:])!=(3,self.size,self.size) or not 3<=images.shape[1]<=12:
            raise ValueError('MLX input must contain 3–12 images at the configured size')
        patch_side = self.size // 14
        patch_count = patch_side ** 2
        token_count = patch_count + 5
        batch,count,_,height,width=images.shape
        images=(images-mx.array([.485,.456,.406],dtype=images.dtype)[None,None,:,None,None])
        images=images/mx.array([.229,.224,.225],dtype=images.dtype)[None,None,:,None,None]
        image=images.reshape(count,3,height,width).transpose(0,2,3,1)
        weight=self.weights['patch_embed.patch_embed.proj.weight'].transpose(0,2,3,1)
        tokens=mx.conv2d(image,weight,stride=14)+self.weights['patch_embed.patch_embed.proj.bias']
        tokens=tokens.reshape(count,-1,1024)
        cls=mx.broadcast_to(self.weights['patch_embed.cls_token'],(count,1,1024))
        tokens=mx.concatenate((cls,tokens),axis=1)+self.weights['patch_embed.pos_embed']
        registers=mx.broadcast_to(self.weights['patch_embed.register_tokens'],(count,4,1024))
        tokens=mx.concatenate((tokens[:,:1],registers,tokens[:,1:]),axis=1)
        for i,block in enumerate(self.blocks['patch_embed.blocks']):
            tokens=block(tokens);mx.async_eval(tokens)
            if progress and i in (0,11,23):progress(f'dino-{i}')
        tokens=mx.fast.layer_norm(tokens,self.weights['patch_embed.norm.weight'],self.weights['patch_embed.norm.bias'],eps=self.patch_norm_eps)
        patches=tokens[:,5:]
        def expand(name):
            first=self.weights[name][0,:1]
            others=mx.broadcast_to(self.weights[name][0,1:2],(count-1,*first.shape[1:]))
            return mx.concatenate((first,others),axis=0)
        tokens=mx.concatenate((expand('camera_token'),expand('register_token'),patches),axis=1)
        pos = rotary_positions(count, self.size)
        results=[]
        for index in range(self.depth):
            frame=self.blocks['frame_blocks'][index](tokens.reshape(count,token_count,1024),pos)
            mx.async_eval(frame)
            global_tokens=self.blocks['global_blocks'][index](frame.reshape(1,count*token_count,1024),pos.reshape(1,count*token_count,2))
            mx.async_eval(global_tokens)
            tokens=global_tokens.reshape(count,token_count,1024)
            if index in self.cached_layers:
                joined=mx.concatenate((frame.reshape(1,count,token_count,1024),tokens.reshape(1,count,token_count,1024)),axis=-1)
                mx.async_eval(joined);results.append(joined)
            else:results.append(None)
            if progress and index in (0,4,11,17,23):progress(f'alternating-{index}')
        # Submit each block promptly so graph construction overlaps GPU work.
        # Finish every queued result before heads consume it or weights release.
        # Precision, tensor algebra and the four retained layers are unchanged.
        mx.synchronize()
        self.calls+=1
        return results,5

    def audit(self):
        blocks={f'{group}.{i}':block.storage() for group,items in self.blocks.items() for i,block in enumerate(items)}
        return {'backend':'mlx-aggregator-pytorch-heads-v1','bits':self.bits,'groupSize':self.group_size,'calls':self.calls,
                'backendVersion':4,'rotaryPositionPolicy':ROTARY_POSITION_POLICY,
                'executionPolicy':EXECUTION_POLICY,
                'linearComputePolicy':self.linear_policy,
                'blocks':blocks,'otherWeightBytes':sum(t.nbytes for t in self.weights.values()),
                'prepacked':self.prepacked,'modelImageSize':self.size,
                'positionInterpolation':'pinned DINO FP32 bicubic, rounded to FP16' if self.size != 518 else 'native learned positions',
                'note':'Prepacked GPU quantized_matmul; calls counted in eager execution; asynchronous block evaluation with final synchronization'}
