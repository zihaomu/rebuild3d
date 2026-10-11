#!/usr/bin/env python3
"""Compare MLX packed affine matmul to both MLX and current MPS FP16."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import mlx.core as mx
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Output must be new')
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(3021)
    report = {'mlx':importlib.metadata.version('mlx'),'torch':torch.__version__,
              'device':mx.device_info(),'repeats':5,'warmups':2,'synchronized':True,'results':[]}
    with torch.inference_mode():
        for m,k,n in [(7,1024,3072),(1374,1024,3072),(9618,1024,3072),(9618,1024,4096),(9618,4096,1024)]:
            weights = (rng.normal(size=(n,k))*.03).astype('float16')
            activations = rng.normal(size=(m,k)).astype('float16')
            w, x = mx.array(weights), mx.array(activations)
            tw, tx = torch.from_numpy(weights).to('mps'), torch.from_numpy(activations).to('mps')
            mx.eval(w,x); torch.mps.synchronize()
            functions = {'mps-fp16':lambda:torch.nn.functional.linear(tx,tw), 'mlx-fp16':lambda:x@w.T}
            packed = {}
            for bits,group in [(8,64),(4,64),(4,128)]:
                label = f'mlx-w{bits}-g{group}'
                q,s,b = mx.quantize(w,group_size=group,bits=bits,mode='affine')
                mx.eval(q,s,b)
                packed[label] = (q,s,b)
                functions[label] = lambda q=q,s=s,b=b,bits=bits,group=group:mx.quantized_matmul(x,q,s,b,group_size=group,bits=bits,mode='affine')

            def run(label):
                if label.startswith('mps'):
                    torch.mps.synchronize(); start=time.perf_counter(); y=functions[label](); torch.mps.synchronize()
                else:
                    mx.synchronize(); start=time.perf_counter(); y=functions[label](); mx.eval(y); mx.synchronize()
                return time.perf_counter()-start, y

            for label in functions:
                for _ in range(2):
                    run(label)
            seconds = {label:[] for label in functions}
            labels = list(functions)
            for iteration in range(5):
                for label in labels[iteration:]+labels[:iteration]:
                    elapsed,y = run(label); seconds[label].append(elapsed)
            reference = run('mps-fp16')[1].cpu().float().numpy()
            errors = {}
            for label in functions:
                _,y = run(label)
                result = y.cpu().float().numpy() if label.startswith('mps') else np.asarray(y,dtype='float32')
                errors[label] = {'relativeRMSE':float(np.sqrt(np.mean((result-reference)**2)/np.mean(reference**2))),
                                 'finite':bool(np.isfinite(result).all())}
            row = {'M':m,'K':k,'N':n,'seconds':seconds,'medianSeconds':{label:statistics.median(t) for label,t in seconds.items()},
                   'errors':errors,'packing':{label:{'dtype':str(arrays[0].dtype),'shape':list(arrays[0].shape),
                     'bytesIncludingScales':sum(z.nbytes for z in arrays)} for label,arrays in packed.items()}}
            report['results'].append(row)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('M','K','N','medianSeconds')}),flush=True)
            del functions,packed,w,x,tw,tx,q,s,b,y
            mx.clear_cache(); torch.mps.empty_cache()


if __name__ == '__main__':
    main()
