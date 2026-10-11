#!/usr/bin/env python3
"""Isolated full inference probe: MLX aggregator, unchanged FP32 MPS output heads.

Accepts verified comparison or offline packed weights. On-the-fly quantization
is only a development comparison; production must use the packed artifact.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from unittest.mock import patch

os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Runtime'))
import mlx.core as mx
import numpy as np
from safetensors import safe_open
import torch
from rebuild3d_worker.infer import prepare, verify_repository, write_json
from rebuild3d_worker.model_weights import digest, kernel_audit, meta_model, verify_weights
from benchmark import footprint
from mlx_aggregator import Aggregator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--bits',type=int,choices=[0,8,4],default=8);p.add_argument('--group-size',type=int,default=64)
    p.add_argument('--heads',choices=['fp32','fp16'],default='fp32',help='Independent precision experiment, never silently enabled')
    p.add_argument('--max-memory-gib',type=float,default=10.5)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    head_dtype=torch.float32 if a.heads=='fp32' else torch.float16
    mx.set_default_device(mx.gpu);mx.set_cache_limit(256*1024**2)
    started=time.monotonic();stop=threading.Event()
    report={'status':'running','stage':'verify','timingsSeconds':{},'scope':'Experimental inference; not app or deployment acceptance'}
    def monitor():
        with (a.output/'resources.jsonl').open('w') as f:
            while not stop.is_set():
                physical=footprint(os.getpid())
                row={'seconds':time.monotonic()-started,'stage':report['stage'],'physicalFootprintBytes':physical,
                     'mpsDriverBytes':torch.mps.driver_allocated_memory(),'mlxActiveBytes':mx.get_active_memory(),
                     'mlxCacheBytes':mx.get_cache_memory(),'mlxPeakActiveBytes':mx.get_peak_memory()}
                f.write(json.dumps(row)+'\n');f.flush()
                if physical is not None and physical>a.max_memory_gib*1024**3:
                    report.update(status='resource-limit',resources=row);write_json(a.output/'report.json',report);os._exit(2)
                stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    @contextmanager
    def timed(name):
        report['stage']=name;mx.synchronize();torch.mps.synchronize();start=time.monotonic()
        try:yield
        finally:
            mx.synchronize();torch.mps.synchronize();report['timingsSeconds'][name]=time.monotonic()-start
    try:
        with timed('verify'):
            verify_repository(a.repo);info=verify_weights(a.weights)
        prepacked=info['format']=='rebuild3d-vggt-mlx'
        if not prepacked and info.get('policy')!='trimmed-fp16':raise ValueError('Probe requires verified FP16 or MLX packed artifact')
        if prepacked and importlib.metadata.version('mlx')!=info['runtimeMLXVersion']:raise ValueError('MLX runtime version mismatch')
        stored_heads16=info.get('headPrecision')=='fp16-parameters-fp32-position-and-activation-v1'
        if stored_heads16 and a.heads!='fp16':raise ValueError('FP16 head artifact requires its explicit FP32-sensitive-op policy')
        records=json.loads((a.dataset/'dataset.json').read_text())['records']
        with timed('prepareInputs'):
            rgb,colors,masks,transforms=prepare(a.dataset,records,518,False)
            # Match current input scaling before crossing frameworks.
            images=torch.from_numpy(rgb.copy()).permute(0,3,1,2).to('mps',torch.float16)[None]/255
            mlx_images=mx.array(images.cpu().numpy())
            mx.eval(mlx_images)
        config={'backendID':'experimental-mlx-aggregator-pytorch-heads','mlx':importlib.metadata.version('mlx'),
                'torch':torch.__version__,'bits':a.bits,'groupSize':a.group_size,'heads':a.heads,'prepacked':prepacked,'baseArtifact':info,
                'photoIDs':[r['id'] for r in records],'sourceSHA256':[r['sourceSHA256'] for r in records],
                'datasetSHA256':digest(a.dataset/'dataset.json'),'modelSize':518,
                'calibration':'No activation fitting; offline weight-only affine algorithm',
                'codeSHA256':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('mlx_aggregator.py'),Path(__file__).with_name('mlx_block.py')]}}
        write_json(a.output/'config.json',config);write_json(a.output/'pixel-transforms.json',transforms)
        with timed('loadPackedWeights' if prepacked else 'loadAndDevelopmentQuantization'):
            model=meta_model(a.repo)
            aggregator=Aggregator(a.weights,model.aggregator,a.bits,a.group_size,weight_info=info)
            model.aggregator=torch.nn.Identity()
            with safe_open(a.weights,framework='pt',device='cpu') as checkpoint:
                for name,parameter in model.named_parameters():
                    value=checkpoint.get_tensor(name)
                    stored_dtype=torch.float16 if stored_heads16 else torch.float32
                    if value.shape!=parameter.shape or value.dtype!=stored_dtype:raise ValueError('Output-head weight mismatch')
                    owner,leaf=name.rsplit('.',1)
                    setattr(model.get_submodule(owner),leaf,torch.nn.Parameter(value.to('mps',head_dtype),requires_grad=False))
            model.eval()
            if a.heads=='fp16':
                # The upstream position helper explicitly returns FP32. Keep
                # grid/trigonometry in FP32 and round only the residual result.
                apply_position=model.depth_head._apply_pos_embed
                model.depth_head._apply_pos_embed=lambda x,*args,**kwargs:apply_position(x.float(),*args,**kwargs).to(x.dtype)
                config['headPrecisionDetails']='FP16 parameters/convolutions; FP32 positional grids and depth/confidence activation'
                write_json(a.output/'config.json',config)
        with timed('aggregator'):
            tokens,patch_start=aggregator(mlx_images,lambda stage:print(stage,flush=True))
        with timed('frameworkTransferAndRelease'):
            tokens=[torch.from_numpy(np.asarray(t).copy()).to('mps',head_dtype) if t is not None else None for t in tokens]
            write_json(a.output/'kernel-audit.json',aggregator.audit())
            del aggregator,mlx_images
            mx.clear_cache()
        with torch.inference_mode():
            with timed('cameraHeadAndReadback'):
                pose=model.camera_head(tokens)[-1].float().cpu()
            with timed('depthHeadAndReadback'):
                from vggt.heads.dpt_head import activate_head
                with patch('vggt.heads.dpt_head.activate_head',lambda out,**kwargs:activate_head(out.float(),**kwargs)):
                    depth,confidence=model.depth_head(tokens,images.to(head_dtype),patch_start,frames_chunk_size=1)
                depth=depth.float().cpu().numpy()[0,...,0];confidence=confidence.float().cpu().numpy()[0]
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        with timed('serializeOutputs'):
            extrinsics,intrinsics=pose_encoding_to_extri_intri(pose,(518,518))
            extrinsics,intrinsics=extrinsics.numpy()[0],intrinsics.numpy()[0]
            if not all(np.isfinite(x).all() for x in (depth,confidence,extrinsics,intrinsics)):raise ValueError('Nonfinite output')
            np.savez_compressed(a.output/'predictions.npz',depth=depth,confidence=confidence,extrinsics=extrinsics,intrinsics=intrinsics,rgb=colors,masks=masks)
            cameras=[{'name':r['name'],'photoID':r['id'],'source':'experimental quantized learned estimate; unverified geometry',
                      'worldToCamera':e.tolist(),'modelIntrinsics':k.tolist(),'workingIntrinsics':(np.linalg.inv(t['workingToModelPixels'])@k).tolist()}
                     for r,e,k,t in zip(records,extrinsics,intrinsics,transforms)]
            write_json(a.output/'cameras.json',{'convention':'OpenCV: right, down, forward; relative scale','cameras':cameras})
        report.update(status='inference-complete',elapsedSeconds=time.monotonic()-started,cameraCount=len(records))
        print(json.dumps(report),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        stop.set();thread.join(timeout=3);write_json(a.output/'report.json',report)


if __name__=='__main__':main()
