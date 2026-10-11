#!/usr/bin/env python3
"""Review actual USDZ renders against original masks and a frozen full model.

All images come from render-review.swift and real input photographs. Numeric
checks describe input consistency, not measured 3D truth or visual acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mask_iou(a, b):
    if a.shape != b.shape or not np.any(a | b):
        raise ValueError('Invalid mask shape or empty silhouette')
    return float(np.count_nonzero(a & b) / np.count_nonzero(a | b))


def paste(sheet, image, x, y, width, height, label):
    image = image.convert('RGB').copy()
    image.thumbnail((width-8, height-28), Image.Resampling.LANCZOS)
    sheet.paste(image, (x+(width-image.width)//2, y+25))
    ImageDraw.Draw(sheet).text((x+5, y+5), label, fill='white')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path); p.add_argument('candidate', type=Path)
    p.add_argument('regions', type=Path); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--renders',default='renders',choices=['renders','renders-cpu'])
    p.add_argument('--output-name',default='review-v4')
    p.add_argument('--candidate-label',default='W8')
    a=p.parse_args(); old,new=a.baseline,a.candidate
    if any(Path(x).name != x for x in (a.output_name,a.candidate_label)):
        raise ValueError('Output names must be simple filenames')
    out=new/a.output_name; out.mkdir(exist_ok=False)
    renderer_records={}
    for name,root in (('baseline',old),('candidate',new)):
        record=root/a.renders/'render-manifest.json'
        renderer_records[name]=read(record) if record.exists() else {'renderer':'SceneKit; see native render logs'}
        if a.renders=='renders-cpu' and renderer_records[name].get('modelSHA256') != sha(root/'bundle/model.usdz'):
            raise ValueError('CPU renders do not match the current USDZ')
    before=read(old/'inputs/dataset.json')['records']; after=read(new/'inputs/dataset.json')['records']
    if [r['sourceSHA256'] for r in before] != [r['sourceSHA256'] for r in after]:
        raise ValueError('Photos or their order changed')
    thresholds=read(a.protocol)['thresholds']
    sheet=Image.new('RGB',(240*len(after),5*345),'#222222')
    views=[]
    for i,(r0,r1) in enumerate(zip(before,after)):
        photo=Image.open(new/'inputs'/r1['image'])
        m0=np.asarray(Image.open(old/'inputs'/r0['mask']).convert('L'))>=128
        m1=np.asarray(Image.open(new/'inputs'/r1['mask']).convert('L'))>=128
        if not np.array_equal(m0,m1):raise ValueError('Prepared photo masks changed')
        silhouettes=[np.asarray(Image.open(path/a.renders/f'view-{i}-mask.png').convert('L'))>=128 for path in (old,new)]
        scores=[mask_iou(m1,m) for m in silhouettes]
        views.append({'photo':r1['name'],'sourceSHA256':r1['sourceSHA256'],
                      'baselinePhotoMaskIoU':scores[0],'candidatePhotoMaskIoU':scores[1],
                      'drop':scores[0]-scores[1],'passed':scores[0]-scores[1]<=thresholds['silhouetteIoUDrop'],
                      'renderMaskAgreement':mask_iou(*silhouettes),'visualVerdict':'pending'})
        images=[photo,*[Image.open(path/a.renders/f'view-{i}-{mode}.png') for mode in ('color','clay') for path in (old,new)]]
        for row,(image,label) in enumerate(zip(images,('Photo','Baseline texture',a.candidate_label+' texture','Baseline clay',a.candidate_label+' clay'))):
            paste(sheet,image,i*240,row*345,240,345,f'{i} {label}')
    sheet.save(out/'all-views.jpg',quality=95)
    regions=read(a.regions)['regions']
    detail=Image.new('RGB',(3*400,len(regions)*320),'#222222')
    for row,region in enumerate(regions):
        i=next(i for i,r in enumerate(after) if r['sourceSHA256']==region['photoSHA256'])
        paths=[new/'inputs'/after[i]['image'],old/a.renders/f'view-{i}-color.png',new/a.renders/f'view-{i}-color.png']
        for col,(path,label) in enumerate(zip(paths,('Photo','Baseline',a.candidate_label))):
            crop=Image.open(path).crop(region['box'])
            crop.save(out/f'{region["id"]}-{label}.png')
            paste(detail,crop,col*400,row*320,400,320,region['id']+' '+label)
    detail.save(out/'fixed-details.jpg',quality=95)
    turntable=Image.new('RGB',(4*320,6*345),'#222222')
    for i in range(12):
        for row,(path,label) in enumerate(((old,'Baseline'),(new,a.candidate_label))):
            paste(turntable,Image.open(path/a.renders/f'turntable-clay-{i:02d}.png'),
                  (i%4)*320,((i//4)*2+row)*345,320,345,f'{i} {label}')
    turntable.save(out/'clay-turntable.jpg',quality=95)
    reports=[read(path/'texture/report.json') for path in (old,new)]
    coverage={key:{'baseline':reports[0][key],'candidate':reports[1][key],
                   'drop':reports[0][key]-reports[1][key]}
              for key in ('photoCoverageTexelFraction','photoSupportedFaceAreaFraction')}
    uv_audits={name:read(path/'texture/uv-audit.json')['surfaceAreaCoverageEstimate']
               for name,path in (('baseline',old),('candidate',new))}
    gates={'silhouettes':all(v['passed'] for v in views),
           'photoTexelCoverage':coverage['photoCoverageTexelFraction']['drop']<=thresholds['photoTexelCoverageDrop']}
    output={'baseline':str(old),'candidate':str(new),'regionsSHA256':sha(a.regions),'views':views,
            'coverage':coverage,'rasterWeightedSurfaceCoverage':uv_audits,'gates':gates,
            'numericGatesPassed':all(gates.values()),'visualAcceptance':'pending','rendering':renderer_records,
            'nativeApplicationAcceptance':'pending; CPU diagnostics cannot prove native display behavior',
            'fixedRegions':[{'id':r['id'],'verdict':'pending'} for r in regions],
            'modelSHA256':{name:sha(path/'bundle/model.usdz') for name,path in (('baseline',old),('candidate',new))},
            'limits':'Each model uses its inferred cameras. Photo consistency is not independent geometry accuracy; all views and fixed details still require visual review.'}
    (out/'metrics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'gates':gates,'coverage':coverage,'output':str(out)}))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
