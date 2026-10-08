"""Measure independent image matches against the fixed approximate surface/cameras.

Matches are observations in two photos, not camera fit targets. No cameras are
modified. Repeated carved patterns can still produce false matches; retain the
matches and overlays and do not interpret these errors as scan accuracy.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from common import camera_project, raster, write_json

p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();cv2.setRNGSeed(42)
records=json.loads((a.inputs/'dataset.json').read_text())['records']
cameras=json.loads((a.candidate/'cameras.json').read_text())['cameras']
g=np.load(a.candidate/'geometry.npz');v=g['vertices'];f=g['faces']
features=[];depths=[];images=[];masks=[]
sift=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.025)
for r,c in zip(records,cameras):
    im=np.array(Image.open(a.inputs/r['image']).convert('RGB').resize((1200,1600),Image.Resampling.LANCZOS))
    mask=np.array(Image.open(a.inputs/r['mask']).convert('L').resize((1200,1600),Image.Resampling.NEAREST))
    mask=cv2.erode((mask>=254).astype('uint8')*255,np.ones((7,7),'uint8'))
    kp,desc=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),mask)
    features.append((np.array([k.pt for k in kp]),desc));images.append(im);masks.append(mask)
    z,ids,b=raster(camera_project(v,np.array(c['worldToCamera']),np.array(c['workingIntrinsics'])),f,1200,1600)
    depths.append(z)
    del ids,b
results=[]
for i,j in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,0)]:
    pi,di=features[i];pj,dj=features[j];matcher=cv2.BFMatcher()
    forward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(di,dj,k=2) if m.distance<.7*n.distance}
    reverse={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(dj,di,k=2) if m.distance<.7*n.distance}
    pairs=np.array([(x,y) for x,y in forward.items() if reverse.get(y)==x],dtype=int).reshape(-1,2);x=pi[pairs[:,0]];y=pj[pairs[:,1]]
    if len(pairs)<8:
        results.append({'source':records[i]['name'],'target':records[j]['name'],'mutualRatioMatches':len(pairs),'status':'insufficient matches; no alignment claim'});continue
    _,inliers=cv2.findFundamentalMat(x,y,cv2.FM_RANSAC,1.5,.999)
    if inliers is None:
        results.append({'source':records[i]['name'],'target':records[j]['name'],'mutualRatioMatches':len(pairs),'status':'robust two-view fit failed; no alignment claim'});continue
    good=inliers.ravel().astype(bool);x=x[good];y=y[good]
    ci,cj=cameras[i],cameras[j];ei=np.array(ci['worldToCamera']);ej=np.array(cj['worldToCamera'])
    ki=np.array(ci['workingIntrinsics']);kj=np.array(cj['workingIntrinsics'])
    depth=depths[i][np.rint(x[:,1]).astype(int),np.rint(x[:,0]).astype(int)]
    good=np.isfinite(depth)&(depth<1e10)&(depth>0);x=x[good];y=y[good];depth=depth[good]
    world=((np.c_[x,np.ones(len(x))]@np.linalg.inv(ki).T)*depth[:,None]-ei[:,3])@ei[:,:3]
    projected=camera_project(world,ej,kj);q=projected[:,:2]
    inside=(q[:,0]>=1)&(q[:,0]<1199)&(q[:,1]>=1)&(q[:,1]<1599)
    idx=np.rint(q).astype(int);idx[:,0]=idx[:,0].clip(0,1199);idx[:,1]=idx[:,1].clip(0,1599)
    expected=depths[j][idx[:,1],idx[:,0]]
    good=inside&(masks[j][idx[:,1],idx[:,0]]>0)&(abs(expected-projected[:,2])<.006)
    x=x[good];y=y[good];q=q[good];world=world[good];delta=q-y;error=np.linalg.norm(delta,axis=1)
    if len(x)==0:
        results.append({'source':records[i]['name'],'target':records[j]['name'],'status':'no jointly visible surface matches'});continue
    np.savez_compressed(a.output/f'pair-{i}-{j}.npz',sourcePixels=x,observedTargetPixels=y,predictedTargetPixels=q,worldPoints=world,residual=delta)
    panel=Image.fromarray(images[j]);draw=ImageDraw.Draw(panel)
    for observed,predicted in zip(y[::max(1,len(y)//120)],q[::max(1,len(y)//120)]):
        xx,yy=observed;draw.ellipse((xx-3,yy-3,xx+3,yy+3),outline='#00ff00',width=2)
        draw.line([tuple(observed),tuple(predicted)],fill='#ff3030',width=2)
    panel.save(a.output/f'pair-{i}-{j}.png')
    item={'source':records[i]['name'],'target':records[j]['name'],'mutualRatioMatches':len(pairs),'fundamentalInliers':int(inliers.sum()),
          'surfaceVisibleMatches':len(x),'medianPixels':float(np.median(error)),'p95Pixels':float(np.percentile(error,95)),
          'medianSignedXY':np.median(delta,axis=0).tolist(),'fineBlendGatePassed':bool(np.median(error)<=2 and np.percentile(error,95)<=5)}
    results.append(item);print(json.dumps(item),flush=True)
write_json(a.output/'report.json',{'method':'SIFT mutual ratio .70, fundamental RANSAC 1.5px; independent matches, no camera fit; mesh ray depth and target visibility',
    'workingSize':[1200,1600],'cameraChanged':False,'pairs':results,'seconds':time.monotonic()-start,
    'limits':'Depth and cameras are estimated. Residuals combine surface, camera and matching errors; no precise multi-view blending gate is assumed.'})
