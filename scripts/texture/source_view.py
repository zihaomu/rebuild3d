"""Prepare a padded display-only source atlas; semantic index maps stay untouched."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from common import write_json

p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);a=p.parse_args();root=a.candidate
ids=np.load(root/'atlas-surface.npz')['originalFaceID'];active=ids>=0
im=np.array(Image.open(root/'texture-source-view.png').convert('RGB'))
dist,nearest=distance_transform_edt(~active,return_indices=True);pad=(~active)&(dist<=5)
im[pad]=im[nearest[0][pad],nearest[1][pad]];Image.fromarray(im).save(root/'texture-source-display.png')
palette=[[125,125,125],[230,65,50],[255,175,20],[80,190,90],[50,190,220],[70,90,230],[185,70,220],[220,120,160]]
legend=Image.new('RGB',(760,500),'#222222');d=ImageDraw.Draw(legend)
d.text((20,15),'TEXTURE SOURCES - all geometry remains inferred',fill='white',font_size=22)
for i,color in enumerate(palette):
    y=60+i*45;d.rectangle((24,y,70,y+28),fill=tuple(color))
    name='Appearance fill / no validated photo sample' if i==0 else f'IMG_{4522+i}.HEIC - photo-derived color'
    d.text((90,y+3),name,fill='white',font_size=19)
d.text((24,448),'Quality index: 1 = edge/oblique; 2 = visible interior (not calibrated truth)',fill='white',font_size=16)
legend.save(root/'texture-source-legend.png')
write_json(root/'texture-source-legend.json',{'codes':[{'index':i,'rgb':c,'meaning':'appearance fill' if i==0 else f'IMG_{4522+i}.HEIC'} for i,c in enumerate(palette)],
 'geometryMeaning':'All geometry inferred; orange/purple geometric classes are separately preserved in provenance.usdz.',
 'displayOnlyPaddingPixels':5,'semanticMap':'texture-source.png','qualityMap':'texture-quality.png'})
