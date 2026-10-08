"""Create a new immutable input revision with documented background exclusions."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
from common import sha,write_json

p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('annotations',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);(a.output/'images').mkdir();(a.output/'masks').mkdir();start=time.monotonic()
data=copy.deepcopy(json.loads((a.inputs/'dataset.json').read_text()));ann=json.loads(a.annotations.read_text());reports=[]
for r in data['records']:
    source=a.inputs/r['mask'];assert sha(source)==r['maskSHA256'];assert sha(a.inputs/r['image'])==r['imageSHA256']
    original=np.array(Image.open(source).convert('L'));mask=original.copy();h,w=mask.shape
    exclude=Image.new('L',(w,h));draw=ImageDraw.Draw(exclude)
    for polygon in ann['backgroundPolygons'][r['name']]:
        draw.polygon([((x+.5)*w/1200-.5,(y+.5)*h/1600-.5) for x,y in polygon],fill=255)
    # Native-resolution review exposed narrow residual strips at manually traced
    # gaps. Reserve an explicit uncertainty band around those exclusions only.
    gap_radius=round(ann.get('manualExclusionMarginWorkingPixels',0)*h/1600)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*gap_radius+1,2*gap_radius+1))
    excluded=cv2.dilate(np.array(exclude),kernel)>0;mask[excluded]=0
    margin=ann.get('perPhotoLowerBoundaryMarginWorkingPixels',{}).get(r['name'],ann['lowerBoundaryMarginWorkingPixels'])
    radius=round(margin*h/1600)
    interior=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    low=round(ann['lowerBoundaryMarginStartsAtWorkingY']*h/1600);mask[low:]=interior[low:]
    sky_count=0;sky=ann.get('instrumentSkyExclusion',{})
    if r['name'] in sky.get('boxes',{}):
        box=np.array(sky['boxes'][r['name']],dtype=float)
        box=np.rint((box+.5)*h/1600-.5).astype(int);x0,y0,x1,y1=box
        im=np.array(Image.open(a.inputs/r['image']).convert('RGB').crop(tuple(box)),dtype='float32')
        blue=(im[:,:,2]>sky['blueRedRatio']*im[:,:,0])&(im[:,:,2]>sky['blueGreenRatio']*im[:,:,1])&(im[:,:,2]-im[:,:,0]>sky['minimumBlueMinusRed'])
        size=2*sky['fullPixelMargin']+1;blue=cv2.dilate(blue.astype('uint8'),np.ones((size,size),'uint8'))>0
        patch=mask[y0:y1,x0:x1];sky_count=int(((patch>=254)&blue).sum());patch[blue]=0
    Image.fromarray(mask).save(a.output/r['mask']);shutil.copy2(a.inputs/r['image'],a.output/r['image'])
    old=r['maskSHA256'];r['parentMaskSHA256']=old;r['maskSHA256']=sha(a.output/r['mask']);r['maskRevision']=3
    r['fullImage']=r['image'];r['fullMask']=r['mask'];r['fullMaskSHA256']=r['maskSHA256']
    r['maskMethod']='Inherited Vision mask intersected with documented background exclusions and conservative lower-boundary sampling margin'
    r['maskAnnotation']='mask-corrections.json';r['reviewStatus']='stage2 manual visual review; conservative sampling, not ground truth'
    reports.append({'photo':r['name'],'parentMaskSHA256':old,'maskSHA256':r['maskSHA256'],
                    'previousCertainForegroundPixels':int((original>=254).sum()),'certainForegroundPixels':int((mask>=254).sum()),
                    'removedCertainPixels':int(((original>=254)&(mask<254)).sum()),'instrumentSkyPixelsRemoved':sky_count,'polygons':len(ann['backgroundPolygons'][r['name']])})
data['parentTextureDatasetSHA256']=sha(a.inputs/'dataset.json');data['maskCorrectionSHA256']=sha(a.annotations)
shutil.copy2(a.annotations,a.output/'mask-corrections.json');write_json(a.output/'dataset.json',data)
write_json(a.output/'mask-revision-report.json',{'photos':reports,'elapsedAutomaticSeconds':time.monotonic()-start,'manualReviewEstimateMinutes':ann['reviewTimeEstimateMinutes'],
    'geometryChanged':False,'limitations':ann['limitations']})
