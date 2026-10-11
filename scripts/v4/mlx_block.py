"""Bounded MLX experiment matching the pinned VGGT Block's inference algebra.

Architecture reference: Meta VGGT layers/block.py, attention.py, mlp.py, rope.py
at a288dd0f14786c93483e45524328726ab7b1b4ce (Apache-2.0 layer sources).
This is not yet a complete or default reconstruction backend.
"""
import math

import mlx.core as mx
import mlx.nn as nn


class Block:
    def __init__(self, weights, config, bits=0, group_size=64):
        self.config, self.bits, self.group_size = config, bits, group_size
        self.weights = dict(weights)
        self.packed = {}
        if config['activation'] != 'GELU' or config['activationApproximation'] != 'none':
            raise ValueError('Unvalidated block activation')
        for name in ('attn.qkv','attn.proj','mlp.fc1','mlp.fc2'):
            if bits:
                if name+'.qweight' in self.weights:
                    q,s,b=(self.weights.pop(name+suffix) for suffix in ('.qweight','.scales','.offsets'))
                    if q.dtype!=mx.uint32 or s.dtype!=mx.float16 or b.dtype!=mx.float16 or s.shape!=b.shape or q.shape[0]!=s.shape[0] or q.shape[1]*32//bits!=s.shape[1]*group_size:
                        raise ValueError('Prepacked MLX Linear layout mismatch')
                    self.packed[name]=(q,s,b)
                else:
                    self.packed[name] = mx.quantize(self.weights.pop(name+'.weight'), group_size=group_size,bits=bits,mode='affine')
        mx.eval(self.weights, self.packed)

    def linear(self, name, x):
        if name in self.packed:
            q,s,b = self.packed[name]
            value = mx.quantized_matmul(x,q,s,b,group_size=self.group_size,bits=self.bits,mode='affine')
            return value+self.weights[name+'.bias'] if name+'.bias' in self.weights else value
        if name+'.bias' in self.weights:
            return mx.addmm(self.weights[name+'.bias'],x,self.weights[name+'.weight'].T)
        return x @ self.weights[name+'.weight'].T

    def norm(self, name, x, epsilon):
        return mx.fast.layer_norm(x,self.weights[name+'.weight'],self.weights[name+'.bias'],eps=epsilon)

    def rope(self, tokens, positions):
        dim = tokens.shape[-1]//2
        inv = 1/(self.config['ropeFrequency'] ** (mx.arange(0,dim,2,dtype=mx.float32)/dim))
        result=[]
        for axis in (0,1):
            angles=(positions[...,axis,None].astype(mx.float32)*inv).astype(tokens.dtype)
            angles=mx.concatenate((angles,angles),axis=-1)[:,None,:,:]
            cos,sin=mx.cos(angles),mx.sin(angles)
            part=tokens[...,axis*dim:(axis+1)*dim]
            rotated=mx.concatenate((-part[...,dim//2:],part[...,:dim//2]),axis=-1)
            result.append(part*cos+rotated*sin)
        return mx.concatenate(result,axis=-1)

    def __call__(self,x,positions=None):
        cfg=self.config;batch,length,width=x.shape;heads=cfg['heads'];head_dim=width//heads
        z=self.norm('norm1',x,cfg['norm1Epsilon'])
        qkv=self.linear('attn.qkv',z).reshape(batch,length,3,heads,head_dim).transpose(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        if cfg['qkNormEpsilon'] is not None:
            q=self.norm('attn.q_norm',q,cfg['qkNormEpsilon'])
            k=self.norm('attn.k_norm',k,cfg['qkNormEpsilon'])
        if cfg['ropeFrequency'] is not None:
            if positions is None:raise ValueError('Missing spatial positions')
            q,k=self.rope(q,positions),self.rope(k,positions)
        attended=mx.fast.scaled_dot_product_attention(q,k,v,scale=head_dim**-.5)
        z=self.linear('attn.proj',attended.transpose(0,2,1,3).reshape(batch,length,width))
        if 'ls1.gamma' in self.weights:z=z*self.weights['ls1.gamma']
        x=x+z
        z=self.linear('mlp.fc1',self.norm('norm2',x,cfg['norm2Epsilon']))
        z=nn.gelu(z.astype(mx.float32)).astype(z.dtype)
        z=self.linear('mlp.fc2',z)
        if 'ls2.gamma' in self.weights:z=z*self.weights['ls2.gamma']
        return x+z

    def storage(self):
        return {'bits':self.bits,'groupSize':self.group_size,'packedLinearCount':len(self.packed),
                'floatBytes':sum(w.nbytes for w in self.weights.values()),
                'packedBytesIncludingScales':sum(t.nbytes for arrays in self.packed.values() for t in arrays),
                'packedDtypes':{k:str(v[0].dtype) for k,v in self.packed.items()},
                'operation':'mlx.core.quantized_matmul' if self.bits else 'mlx.core.addmm'}
