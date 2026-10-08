"""Held-out check of bounded pose-only refinement; diagnostic, never auto-adopted."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from common import camera_project,write_json

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('cameras',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);cams=json.loads(a.cameras.read_text())['cameras'];results=[]
for path in sorted(a.audit.glob('pair-*.npz')):
    _,src,dst=path.stem.split('-');dst=int(dst);d=np.load(path);world=d['worldPoints'];observed=d['observedTargetPixels']
    if len(world)<20:
        results.append({'target':dst,'status':'too few independent matches for fit and holdout'});continue
    # Frozen spatial split: every third point sorted by y then x is never fitted.
    order=np.lexsort((observed[:,0],observed[:,1]));test=order[::3];fit=np.setdiff1d(order,test)
    c=cams[dst];ext=np.array(c['worldToCamera']);k=np.array(c['workingIntrinsics'])
    camera_world=world@ext[:,:3].T+ext[:,3];translation_limit=float(np.median(camera_world[:,2])*.01)
    def pose(x):
        r=cv2.Rodrigues(x[:3])[0];return np.c_[r@ext[:,:3],r@ext[:,3]+x[3:]]
    def residual(x):
        data=(camera_project(world[fit],pose(x),k)[:,:2]-observed[fit]).ravel()
        return np.r_[data,x[:3]/.01,x[3:]/(translation_limit*.5)]
    limit=np.r_[np.full(3,np.deg2rad(1.)),np.full(3,translation_limit)]
    result=least_squares(residual,np.zeros(6),bounds=(-limit,limit),loss='soft_l1',f_scale=2,diff_step=.001,
                         jac='3-point',xtol=1e-7,ftol=1e-7,gtol=1e-7,max_nfev=100)
    before=np.linalg.norm(camera_project(world,ext,k)[:,:2]-observed,axis=1)
    after=np.linalg.norm(camera_project(world,pose(result.x),k)[:,:2]-observed,axis=1)
    stats=lambda e,idx:{'median':float(np.median(e[idx])),'p95':float(np.percentile(e[idx],95)),'count':len(idx)}
    item={'target':dst,'sourcePair':path.name,'trainBefore':stats(before,fit),'trainAfter':stats(after,fit),
          'holdoutBefore':stats(before,test),'holdoutAfter':stats(after,test),'parameters':result.x.tolist(),
          'candidateWorldToCamera':pose(result.x).tolist(),'fineBlendGatePassed':bool(np.median(after[test])<=2 and np.percentile(after[test],95)<=5),
          'adopted':False}
    results.append(item);print(json.dumps(item),flush=True)
write_json(a.output/'report.json',{'method':'bounded SE3 pose adjustment only; <=1 degree per rotation axis and <=1% median depth per translation axis; robust fit with pose prior',
 'split':'sort y/x, every third observation held out; no held-out point fitted','results':results,
 'decision':'Diagnostic candidates only. Keep original cameras until independent coverage and full-model consistency justify adoption; no texture warp or lens parameters.'})
