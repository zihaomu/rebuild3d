#!/usr/bin/env python3
"""Offline, streaming MLX affine packing from the verified FP16 comparison file."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Runtime'))
import mlx.core as mx
import numpy as np
from safetensors import safe_open
from rebuild3d_worker.infer import verify_repository
from rebuild3d_worker.model_weights import METADATA_KEY, digest, make_recipe, meta_model, verify_weights
from rebuild3d_worker.mlx_block import FP16_LINEAR_POLICY, FP32_LINEAR_POLICY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--bits',type=int,choices=[8,4],default=8);p.add_argument('--group-size',type=int,choices=[32,64,128],default=64)
    p.add_argument('--heads',choices=['fp32','fp16'],default='fp32')
    p.add_argument('--linear-compute-policy',choices=[FP16_LINEAR_POLICY,FP32_LINEAR_POLICY],default=FP16_LINEAR_POLICY,
                   help='Manifest-bound runtime arithmetic; packed tensors remain uint32/FP16')
    a=p.parse_args();start=time.monotonic()
    if a.linear_compute_policy==FP32_LINEAR_POLICY and (a.bits!=8 or a.group_size!=32):
        raise ValueError('FP32 packed GEMM policy requires W8 group 32')
    if importlib.metadata.version('mlx')!='0.32.2':raise ValueError('Pin MLX 0.32.2 for this format')
    verify_repository(a.repo);base=verify_weights(a.source)
    if base.get('policy')!='trimmed-fp16':raise ValueError('Expected the verified FP16 comparison file')
    model=meta_model(a.repo)
    recipe=make_recipe(model,'w8' if a.bits==8 else 'w4',0 if a.bits==8 else a.group_size)
    recipe.update(format='rebuild3d-vggt-mlx',backendID='mlx-aggregator-pytorch-heads-v1',runtimeMLXVersion='0.32.2',
                  policy=f'mlx-w{a.bits}-g{a.group_size}',convertedFromSHA256=base['derivedWeightSHA256'],
                  converterSHA256=digest(Path(__file__)),weightLicense='CC-BY-NC-4.0; unchanged from original checkpoint')
    recipe['headPrecision']='fp32' if a.heads=='fp32' else 'fp16-parameters-fp32-position-and-activation-v1'
    recipe['linearComputePolicy']=a.linear_compute_policy
    if a.heads=='fp16':
        for name in recipe['modulePrecisionMap']:
            if name.startswith(('camera_head.','depth_head.')):recipe['modulePrecisionMap'][name]='FP16'
    for module in recipe['quantizedModules'].values():
        module.update(packing='mlx-affine-uint32-v1',groupSize=a.group_size)
    layout={}
    for name,param in model.named_parameters():
        module=recipe['quantizedModules'].get(name[:-7]) if name.endswith('.weight') else None
        if module:
            n,k=param.shape
            if k%a.group_size:raise ValueError('Group must divide the weight width')
            layout[name[:-7]+'.qweight']=('U32',(n,k*a.bits//32),4)
            for suffix in ('.scales','.offsets'):layout[name[:-7]+suffix]=('F16',(n,k//a.group_size),2)
        else:
            half=name.startswith('aggregator.') or a.heads=='fp16'
            layout[name]=('F16' if half else 'F32',tuple(param.shape),2 if half else 4)
    header={'__metadata__':{METADATA_KEY:json.dumps(recipe,sort_keys=True,separators=(',',':'))}};offset=0
    for name,(dtype,shape,width) in layout.items():
        end=offset+math.prod(shape)*width;header[name]={'dtype':dtype,'shape':shape,'data_offsets':[offset,end]};offset=end
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    a.output.mkdir(parents=True,exist_ok=False);destination=a.output/'vggt.safetensors';partial=destination.with_suffix('.partial')
    # Keep the exact converter implementation with exploratory artifacts; later
    # source edits must not make the recorded conversion impossible to inspect.
    archive=a.output/'converter-sources'
    project_root=Path(__file__).resolve().parents[2]
    for source_path in [Path(__file__).resolve(),project_root/'Runtime/rebuild3d_worker/model_weights.py',project_root/'Runtime/rebuild3d_worker/quantization.py',project_root/'Runtime/rebuild3d_worker/mlx_block.py']:
        saved=archive/source_path.relative_to(project_root);saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes(source_path.read_bytes())
    mx.set_default_device(mx.gpu);mx.set_cache_limit(64*1024**2)
    tensor_hashes={}
    import hashlib
    with safe_open(a.source,framework='np') as source,partial.open('xb') as stream:
        stream.write(struct.pack('<Q',len(encoded)));stream.write(encoded);data_start=stream.tell()
        for index,(name,param) in enumerate(model.named_parameters()):
            value=source.get_tensor(name)
            if tuple(value.shape)!=tuple(param.shape) or not np.isfinite(value).all():raise ValueError('Invalid source tensor')
            module=recipe['quantizedModules'].get(name[:-7]) if name.endswith('.weight') else None
            if module:
                w=mx.array(value);q,s,b=mx.quantize(w,bits=a.bits,group_size=a.group_size,mode='affine');mx.eval(q,s,b)
                tensors=[(name[:-7]+suffix,np.asarray(t)) for suffix,t in zip(('.qweight','.scales','.offsets'),(q,s,b))]
            else:
                value=value.astype('float16' if layout[name][0]=='F16' else 'float32')
                if not np.isfinite(value).all():raise ValueError('Nonfinite converted parameter: '+name)
                tensors=[(name,value)]
            for key,array in tensors:
                entry=header[key]
                if list(array.shape)!=list(entry['shape']) or stream.tell()-data_start!=entry['data_offsets'][0]:raise ValueError('Layout mismatch')
                data=array.tobytes()
                if len(data)!=entry['data_offsets'][1]-entry['data_offsets'][0]:raise ValueError('Dtype/length mismatch')
                stream.write(data);tensor_hashes[key]=hashlib.sha256(data).hexdigest()
            if index%200==0:print('Packed',index,name,flush=True)
        stream.flush();os.fsync(stream.fileno())
    partial.rename(destination)
    with safe_open(destination,framework='np') as packed:
        if set(packed.keys())!=set(layout):raise ValueError('Written tensor list mismatch')
        for name in packed.keys():
            if hashlib.sha256(packed.get_tensor(name).tobytes()).hexdigest()!=tensor_hashes[name]:raise ValueError('Readback mismatch: '+name)
    result={'recipe':recipe,'weightBytes':destination.stat().st_size,'derivedWeightSHA256':digest(destination),
            'verifiedTensorCount':len(tensor_hashes),'conversionAndVerificationSeconds':time.monotonic()-start}
    destination.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    verify_weights(destination)
    print(json.dumps({k:v for k,v in result.items() if k!='recipe'}),flush=True)


if __name__=='__main__':main()
