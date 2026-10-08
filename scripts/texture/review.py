"""Compare actual rendered assets at frozen original-photo ROIs; never auto-accept visual quality."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
from common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('originals',type=Path);a=p.parse_args()
    baseline_repeat=a.candidate.resolve()==a.baseline.resolve()
    candidate_label='stage1-repeat' if baseline_repeat else 'stage2'
    out=a.candidate/'review';out.mkdir(exist_ok=baseline_repeat)
    regions=json.loads((a.baseline/'review-regions.json').read_text())['regions']
    results=[];sheets=[Image.new('RGB',(1080,960),'#222222') for _ in range(2)]
    for i,r in enumerate(regions):
        view=int(Path(r['photo']).stem.split('_')[-1])-4523
        images=[Image.open(a.originals/(Path(r['photo']).stem+'.jpg')),
                Image.open(a.baseline/'renders'/f'view-{view}-color.png'),
                Image.open(a.candidate/'renders'/f'view-{view}-color.png')]
        crops=[im.convert('RGB').crop(r['box']) for im in images]
        for j,(im,label) in enumerate(zip(crops,['original','stage1',candidate_label])):
            im.save(out/f'{r["id"]}-{label}.png')
            sheet=sheets[i//4];x,y=j*360,(i%4)*240
            ImageDraw.Draw(sheet).text((x+10,y+5),r['id']+' / '+label,fill='white')
            sheet.paste(im,(x+(350-im.width)//2,y+40))
        arrays=[np.asarray(im) for im in crops]
        scores=[float(structural_similarity(arrays[0],v,channel_axis=2,data_range=255)) for v in arrays[1:]]
        results.append({'region':r['id'],'photo':r['photo'],'box':r['box'],'ssimStage1':scores[0],'ssimCandidate':scores[1],
                        'visualVerdict':'pending; SSIM alone is not acceptance'})
    for i,sheet in enumerate(sheets):sheet.save(out/f'detail-comparison-{i}.png')
    views=[]
    for i in range(7):
        original=Image.open(a.originals/f'IMG_{4523+i}.jpg').convert('RGB')
        old=Image.open(a.baseline/'renders'/f'view-{i}-color.png').convert('RGB')
        new=Image.open(a.candidate/'renders'/f'view-{i}-color.png').convert('RGB')
        panel=Image.new('RGB',(1200,560),'#222222')
        for j,im in enumerate([original,old,new]):
            panel.paste(im.resize((400,533),Image.Resampling.LANCZOS),(j*400,27))
            ImageDraw.Draw(panel).text((j*400+8,6),['Original','Stage 1',candidate_label][j],fill='white')
        panel.save(out/f'view-{i}-comparison.jpg')
        b=np.array(Image.open(a.baseline/'renders'/f'view-{i}-mask.png').convert('L'))>127
        c=np.array(Image.open(a.candidate/'renders'/f'view-{i}-mask.png').convert('L'))>127
        views.append({'view':i,'maskAgreementWithStage1':float((b&c).sum()/(b|c).sum())})
    write_json(out/'metrics.json',{'candidateLabel':candidate_label,'regions':results,'views':views,'interpretation':'Same input views reused for texture; input consistency only, not independent 3D accuracy.'})
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
