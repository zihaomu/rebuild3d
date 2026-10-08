"""Produce native-resolution edge crops for reviewing the inherited foreground masks."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from common import sha,write_json

p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);records=json.loads((a.inputs/'dataset.json').read_text())['records'];reports=[]
for r in records:
    assert sha(r['sourcePath'])==r['sourceSHA256']
    assert sha(a.inputs/r['image'])==r['imageSHA256'];assert sha(a.inputs/r['mask'])==r['maskSHA256']
    im=np.array(Image.open(a.inputs/r['image']).convert('RGB'));mask=np.array(Image.open(a.inputs/r['mask']).convert('L'))>=254
    y,x=np.where(mask);left,right,top,bottom=int(x.min()),int(x.max()),int(y.min()),int(y.max())
    topx=int(np.median(x[y<top+32]));upper=y<top+(bottom-top)*.65;rightx=int(x[upper].max())
    righty=int(np.median(y[upper & (x>rightx-16)]));bottomx=int(np.median(x[y>bottom-32]))
    centers=[(topx,top+80),(rightx-80,righty),(bottomx,bottom-80)]
    panel=Image.new('RGB',(384*3,420),'#222222');boxes=[]
    for i,(cx,cy) in enumerate(centers):
        xx=int(np.clip(cx-192,0,im.shape[1]-384));yy=int(np.clip(cy-192,0,im.shape[0]-384))
        patch=im[yy:yy+384,xx:xx+384].copy();m=mask[yy:yy+384,xx:xx+384].astype('uint8')
        edge=cv2.dilate(m,np.ones((3,3),'uint8'))!=cv2.erode(m,np.ones((3,3),'uint8'))
        patch[edge]=[255,45,70];panel.paste(Image.fromarray(patch),(384*i,36))
        ImageDraw.Draw(panel).text((384*i+6,10),r['name']+' '+['top','upper-side','bottom'][i]+' / full-res 1:1',fill='white')
        boxes.append([xx,yy,xx+384,yy+384])
    panel.save(a.output/(Path(r['name']).stem+'-edges.png'))
    reports.append({'photo':r['name'],'sourceSHA256':r['sourceSHA256'],'maskSHA256':r['maskSHA256'],
                    'size':[im.shape[1],im.shape[0]],'fullResolutionCropBoxes':boxes,'review':'pending visual inspection'})
write_json(a.output/'mask-audit.json',{'kind':'inherited-mask-high-resolution-edge-review','photos':reports,
    'limitations':'Three reproducible edge samples per image, not an exhaustive pixel-level ground-truth annotation.'})
