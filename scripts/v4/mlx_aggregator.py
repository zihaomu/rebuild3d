"""Isolated 518px MLX VGGT aggregator experiment; not an application backend yet.

Matches pinned Meta VGGT aggregator/DINO ordering, including separate first-view
tokens and the four cached layers. Output heads stay in their current PyTorch FP32.
"""
import mlx.core as mx
from safetensors import safe_open

from mlx_block import Block


class Aggregator:
    def __init__(self, weights, meta, bits=0, group_size=64, weight_info=None):
        self.blocks={};self.weights={};self.bits=bits;self.group_size=group_size
        self.block_configs={};self.calls=0
        self.prepacked=weight_info is not None and weight_info['format']=='rebuild3d-vggt-mlx'
        with safe_open(weights,framework='np') as checkpoint:
            available={k[11:] for k in checkpoint.keys() if k.startswith('aggregator.')}
            expected={k for k,_ in meta.named_parameters()}
            if self.prepacked:
                import torch
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
                    group_blocks.append(Block(tensors,config,bits,group_size))
                    self.block_configs[prefix[:-1]]=config
                    available.difference_update(names)
                    del tensors
                self.blocks[group]=group_blocks
            for name in available:
                self.weights[name]=mx.array(checkpoint.get_tensor('aggregator.'+name))
            mx.eval(self.weights)
        self.patch_norm_eps=meta.patch_embed.norm.eps
        self.cached_layers=set(meta.cached_layer_indices)
        self.depth=meta.depth

    def __call__(self,images,progress=None):
        # Prototype deliberately rejects unsupported geometry rather than
        # silently changing preprocessing or interpolating position embeddings.
        if images.ndim!=5 or images.shape[0]!=1 or tuple(images.shape[2:])!=(3,518,518) or not 3<=images.shape[1]<=12:
            raise ValueError('Experimental MLX aggregator supports 3–12 images at 518px only')
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
            tokens=block(tokens);mx.eval(tokens)
            if progress and i in (0,11,23):progress(f'dino-{i}')
        tokens=mx.fast.layer_norm(tokens,self.weights['patch_embed.norm.weight'],self.weights['patch_embed.norm.bias'],eps=self.patch_norm_eps)
        patches=tokens[:,5:]
        def expand(name):
            first=self.weights[name][0,:1]
            others=mx.broadcast_to(self.weights[name][0,1:2],(count-1,*first.shape[1:]))
            return mx.concatenate((first,others),axis=0)
        tokens=mx.concatenate((expand('camera_token'),expand('register_token'),patches),axis=1)
        grid=mx.stack(mx.meshgrid(mx.arange(37),mx.arange(37),indexing='ij'),axis=-1).reshape(1,1369,2)+1
        pos=mx.concatenate((mx.zeros((count,5,2),dtype=mx.int32),mx.broadcast_to(grid,(count,1369,2))),axis=1)
        results=[]
        for index in range(self.depth):
            frame=self.blocks['frame_blocks'][index](tokens.reshape(count,1374,1024),pos)
            mx.eval(frame)
            global_tokens=self.blocks['global_blocks'][index](frame.reshape(1,count*1374,1024),pos.reshape(1,count*1374,2))
            mx.eval(global_tokens)
            tokens=global_tokens.reshape(count,1374,1024)
            if index in self.cached_layers:
                joined=mx.concatenate((frame.reshape(1,count,1374,1024),tokens.reshape(1,count,1374,1024)),axis=-1)
                mx.eval(joined);results.append(joined)
            else:results.append(None)
            if progress and index in (0,4,11,17,23):progress(f'alternating-{index}')
        self.calls+=1
        return results,5

    def audit(self):
        blocks={f'{group}.{i}':block.storage() for group,items in self.blocks.items() for i,block in enumerate(items)}
        return {'backend':'experimental-mlx-aggregator','bits':self.bits,'groupSize':self.group_size,'calls':self.calls,
                'blocks':blocks,'otherWeightBytes':sum(t.nbytes for t in self.weights.values()),
                'prepacked':self.prepacked,
                'note':'Loaded directly from prepacked artifact' if self.prepacked else 'Quantized during development prototype loading; not a production prepacked artifact'}
