#!/usr/bin/env python3
"""Replay real VGGT layer activations through MPS and MLX FP16/W8/W4."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Runtime'))
import mlx.core as mx
import numpy as np
from safetensors.torch import load_file
import torch
from rebuild3d_worker.infer import verify_repository
from rebuild3d_worker.model_weights import digest, meta_model
from mlx_block import Block


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--compile',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('New output required')
    verify_repository(a.repo); model=meta_model(a.repo)
    mx.set_default_device(mx.gpu)
    results={'scope':'Bounded block check on real captured photo activations; not whole-model acceptance',
             'mlx':importlib.metadata.version('mlx'),'torch':torch.__version__,'device':mx.device_info(),'compiled':a.compile,
             'results':[]}
    with torch.inference_mode():
        for root in sorted(a.capture.iterdir()):
            if not root.is_dir():continue
            config=json.loads((root/'capture.json').read_text())
            if digest(root/'activations.npz')!=config['activationFileSHA256'] or digest(root/'weights.safetensors')!=config['blockWeightSHA256']:
                raise ValueError('Capture changed')
            with np.load(root/'activations.npz') as f: arrays={k:f[k] for k in f.files}
            block=model.get_submodule(config['name'])
            weights=load_file(root/'weights.safetensors')
            block.load_state_dict(weights,assign=True);block=block.to('mps').eval()
            tx=torch.from_numpy(arrays['input']).to('mps');tp=torch.from_numpy(arrays['positions']).to('mps') if 'positions' in arrays else None
            x=mx.array(arrays['input']);pos=mx.array(arrays['positions']) if 'positions' in arrays else None
            mx.eval(x,pos);torch.mps.synchronize()
            variants={'mlx-fp16':Block({k:mx.array(v.numpy()) for k,v in weights.items()},config)}
            for bits,group in [(8,64),(4,64),(4,128)]:
                variants[f'mlx-w{bits}-g{group}']=Block({k:mx.array(v.numpy()) for k,v in weights.items()},config,bits,group)
            compiled={k:mx.compile(lambda x,pos,v=v:v(x,pos)) for k,v in variants.items()} if a.compile else variants
            funcs={'mps-fp16':lambda:block(tx,pos=tp) if tp is not None else block(tx),
                   **{k:lambda v=v:v(x,pos) for k,v in compiled.items()}}
            def run(label):
                if label=='mps-fp16':
                    torch.mps.synchronize();start=time.perf_counter();out=funcs[label]();torch.mps.synchronize()
                else:
                    mx.synchronize();start=time.perf_counter();out=funcs[label]();mx.eval(out);mx.synchronize()
                return time.perf_counter()-start,out
            first_calls={}
            for label in funcs:
                first_calls[label]=run(label)[0]
                for _ in range(2):run(label)
            seconds={k:[] for k in funcs};labels=list(funcs)
            for i in range(5):
                for label in labels[i:]+labels[:i]:
                    elapsed,out=run(label);seconds[label].append(elapsed)
            ref=arrays['output'].astype('float32'); errors={}
            for label in funcs:
                _,out=run(label)
                candidate=out.cpu().float().numpy() if label=='mps-fp16' else np.asarray(out,dtype='float32')
                delta=candidate-ref
                errors[label]={'relativeRMSE':float(np.sqrt(np.mean(delta**2)/np.mean(ref**2))),
                               'maxAbsolute':float(np.abs(delta).max()),'finite':bool(np.isfinite(candidate).all())}
            row={'name':config['name'],'capture':str(root),'inputShape':config['inputShape'],'seconds':seconds,
                 'medianSeconds':{k:statistics.median(t) for k,t in seconds.items()},'errors':errors,
                 'storage':{k:v.storage() for k,v in variants.items()},'firstCallSeconds':first_calls}
            results['results'].append(row);a.output.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('name','inputShape','medianSeconds','errors')}),flush=True)
            # Replace the block in the skeleton so earlier GPU weights can be freed.
            owner,leaf=config['name'].rsplit('.',1);setattr(model.get_submodule(owner),leaf,torch.nn.Identity())
            del funcs,variants,compiled,weights,block,tx,tp,x,pos,out
            mx.clear_cache();torch.mps.empty_cache()


if __name__=='__main__':main()
