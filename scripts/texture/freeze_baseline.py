"""Freeze the prior delivery and detail ROIs before creating new textures."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
from common import sha, write_json, camera_project, raster


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage1',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    dataset=json.loads((a.stage1/'input-v3/dataset.json').read_text())
    run=a.stage1/'reproduction-01'
    paths=[run/'inference/cameras.json',run/'inference/pixel-transforms.json',run/'fusion/face-provenance.npz',
           run/'fusion/report.json',a.stage1/'delivery/Seven-photo-statue.usdz',run/'fusion/usdz/model-region-0.png',
           run/'fusion/usdz/model-region-1.png',a.stage1/'delivery/Seven-photo-statue.rebuild3d/project.json']
    hashes={str(path):sha(path) for path in paths}
    for record in dataset['records']:
        assert sha(record['sourcePath'])==record['sourceSHA256']
    cameras=json.loads((run/'inference/cameras.json').read_text())
    for camera,record in zip(cameras['cameras'],dataset['records']):
        camera['workingSize']=[record['workingWidth'],record['workingHeight']]
        camera['uprightSize']=[record['uprightWidth'],record['uprightHeight']]
    write_json(a.output/'cameras.json',cameras)
    regions=json.loads(Path(__file__).with_name('review-regions.json').read_text())
    geometry=np.load(run/'fusion/face-provenance.npz')
    mappings={}; rendered={}
    sheet=Image.new('RGB',(4*340,2*350),'#222222'); draw=ImageDraw.Draw(sheet)
    for n,region in enumerate(regions['regions']):
        i=next(i for i,c in enumerate(cameras['cameras']) if c['name']==region['photo'])
        if i not in rendered:
            c=cameras['cameras'][i]
            _,ids,_=raster(camera_project(geometry['verticesBeforeDisplay'],np.array(c['worldToCamera']),np.array(c['workingIntrinsics'])),geometry['faces'],1200,1600)
            rendered[i]=ids
        x0,y0,x1,y1=region['box']
        visible=np.unique(rendered[i][y0:y1,x0:x1]); visible=visible[visible>=0]
        mappings[region['id']]=visible
        region['surfaceFaceCount']=len(visible)
        region['sourceSHA256']=dataset['records'][i]['sourceSHA256']
        im=Image.open(a.stage1/'input-v3'/dataset['records'][i]['image']).crop(region['box'])
        im.save(a.output/(region['id']+'-original-working.png'))
        im.thumbnail((330,310))
        x,y=(n%4)*340,(n//4)*350
        sheet.paste(im,(x+(330-im.width)//2,y+30)); draw.text((x+8,y+5),region['id']+' / '+region['photo'],fill='white')
    np.savez_compressed(a.output/'review-region-faces.npz',**mappings)
    write_json(a.output/'review-regions.json',regions)
    sheet.save(a.output/'original-regions.jpg')
    write_json(a.output/'baseline.json',{'createdAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'codeCommit':'5889a6472131bcedbf16a20a423d4c8576b26ff3','hashes':hashes,
        'sourcePhotos':[{k:r[k] for k in ['name','sourceSHA256']} for r in dataset['records']],
        'roiManifestSHA256':sha(a.output/'review-regions.json'),
        'renderSettings':{'workingSize':[1200,1600],'pixelConvention':'integer pixel centers; explicit frame; projection uses cx+0.5,cy+0.5',
                          'modes':['unlit-photo-color','fixed-light-clay','mask'],'turntableViews':12,'turntableSize':[960,960]},
        'geometryUnchanged':True,'status':'frozen-before-any-stage2-texture'})
    print(a.output,flush=True)


if __name__=='__main__': main()
