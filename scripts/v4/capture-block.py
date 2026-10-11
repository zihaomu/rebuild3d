#!/usr/bin/env python3
"""Capture real photo activations for a bounded MLX port check, locally only."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Runtime'))
import numpy as np
from safetensors.torch import save_file
import torch
from rebuild3d_worker.infer import prepare, verify_repository
from rebuild3d_worker.model_weights import digest, load_model, verify_weights


class Captured(Exception):
    pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    verify_repository(a.repo);info=verify_weights(a.weights)
    model,_=load_model(a.repo,a.weights,'mps',info=info)
    records=json.loads((a.dataset/'dataset.json').read_text())['records']
    rgb,_,_,_=prepare(a.dataset,records,518,False)
    image=torch.from_numpy(rgb.copy()).permute(0,3,1,2).to('mps',torch.float16)[None]/255
    names=['aggregator.patch_embed.blocks.0','aggregator.frame_blocks.0','aggregator.global_blocks.0']
    captured=[]
    def hook(name):
        def save(module,inputs,kwargs,output):
            root=a.output/name;root.mkdir()
            arrays={'input':inputs[0].cpu().numpy(),'output':output.cpu().numpy()}
            if kwargs.get('pos') is not None:
                arrays['positions']=kwargs['pos'].cpu().numpy()
            np.savez(root/'activations.npz',**arrays)
            save_file({key:weight.detach().cpu().contiguous() for key,weight in module.state_dict().items()},root/'weights.safetensors')
            config={'name':name,'heads':module.attn.num_heads,'norm1Epsilon':module.norm1.eps,'norm2Epsilon':module.norm2.eps,
                    'qkNormEpsilon':getattr(module.attn.q_norm,'eps',None),'ropeFrequency':getattr(module.attn.rope,'base_frequency',None),
                    'activation':type(module.mlp.act).__name__,'activationApproximation':getattr(module.mlp.act,'approximate',None),
                    'photoIDs':[r['id'] for r in records],'datasetSHA256':digest(a.dataset/'dataset.json'),
                    'weightSHA256':info['derivedWeightSHA256'],'inputShape':list(arrays['input'].shape),
                    'activationFileSHA256':digest(root/'activations.npz'),'blockWeightSHA256':digest(root/'weights.safetensors')}
            (root/'capture.json').write_text(json.dumps(config,indent=2)+'\n')
            captured.append(name);print(json.dumps({'block':name,'shape':config['inputShape']}),flush=True)
            if name==names[-1]:
                raise Captured()
        return save
    handles=[model.get_submodule(name).register_forward_hook(hook(name),with_kwargs=True) for name in names]
    try:
        with torch.inference_mode():
            model.aggregator(image)
    except Captured:
        if captured!=names:
            raise ValueError('Unexpected capture order')
    finally:
        for handle in handles:handle.remove()
    if captured!=names:
        raise ValueError('Missing block activations')


if __name__=='__main__':main()
