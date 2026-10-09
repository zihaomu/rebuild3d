"""Bake full-resolution photographs through a fixed mesh UV atlas, retaining source maps."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
import trimesh

from .common import Budget, camera_project, full_intrinsic, raster, sha, write_json


def sample(image, xy, nearest=False):
    x,y=xy.T
    if nearest:
        return image[np.clip(np.rint(y).astype(int),0,image.shape[0]-1),np.clip(np.rint(x).astype(int),0,image.shape[1]-1)]
    x=np.clip(x,0,image.shape[1]-1.001); y=np.clip(y,0,image.shape[0]-1.001)
    xx=x.astype(int); yy=y.astype(int); dx=x-xx; dy=y-yy
    if image.ndim==3: dx=dx[:,None]; dy=dy[:,None]
    return ((image[yy,xx]*(1-dx)+image[yy,xx+1]*dx)*(1-dy)+
            (image[yy+1,xx]*(1-dx)+image[yy+1,xx+1]*dx)*dy)


def linear(rgb):
    c=np.asarray(rgb)/255.
    return np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)


def encode(rgb):
    c=np.maximum(rgb,0)
    return 255*np.where(c<=.0031308,12.92*c,1.055*c**(1/2.4)-.055)


def valid_projection(points, camera, zbuffer, distance, tolerance):
    p=camera_project(points,camera['ext'],camera['workK'])
    xy,z=p[:,:2],p[:,2]
    height,width=distance.shape
    inside=(z>0)&(xy[:,0]>=1)&(xy[:,0]<width-2)&(xy[:,1]>=1)&(xy[:,1]<height-2)
    md=sample(distance,xy,True)
    dz=abs(z-sample(zbuffer,xy,True))
    valid=inside & (md>0) & (dz<tolerance+z/camera['workK'][0,0]*1.5)
    return valid,md,dz


def main():
    p=argparse.ArgumentParser(); p.add_argument('inputs',type=Path); p.add_argument('run',type=Path)
    p.add_argument('uv',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--smooth',type=int,default=0); p.add_argument('--exposure',action='store_true')
    p.add_argument('--selection',choices=['geometry','quality'],default='geometry')
    p.add_argument('--quality-scale',type=float,default=0,help='Foreground-normalized exposure-quality blur in working-image pixels')
    p.add_argument('--full-mask',action='store_true');p.add_argument('--safe-fill',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); budget=Budget(a.output)
    dataset=json.loads((a.inputs/'dataset.json').read_text()); records=dataset['records']
    run=a.run; g=np.load(run/'fusion/face-provenance.npz')
    vertices=g['verticesBeforeDisplay']; faces=g['faces']; completion=g['completion']
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    centers=mesh.triangles_center; normals=mesh.face_normals
    previous=trimesh.load(run/'fusion/model.ply',process=False)
    # Color fallback remains an explicitly marked coarse prior, never a new photograph observation.
    assert np.array_equal(previous.faces,faces)
    fallback=previous.visual.vertex_colors[:,:3][faces].mean(axis=1).astype('uint8')
    original_cameras=json.loads((run/'inference/cameras.json').read_text())['cameras']
    transforms=json.loads((run/'inference/pixel-transforms.json').read_text())
    photo_count=len(records)
    assert len(original_cameras)==len(transforms)==photo_count
    cameras=[]; buffers=[]; scores=np.zeros((len(faces),photo_count),dtype='float32')
    distances=[]; colors=[];quality_luma=[];angles=np.zeros_like(scores);density=np.zeros_like(scores)
    for i,(record,c,t) in enumerate(zip(records,original_cameras,transforms)):
        assert sha(a.inputs/record['image'])==record['workingSHA256']
        assert sha(a.inputs/record['fullImage'])==record['fullImageSHA256']
        assert sha(a.inputs/record['fullMask'])==record['fullMaskSHA256']
        assert c['photoID']==t['photoID']==record['id'], 'Camera identity mismatch'
        assert sha(a.inputs/record['mask'])==record['maskSHA256']
        assert sha(record['sourcePath'])==record['sourceSHA256']
        k=full_intrinsic(record,np.array(c['modelIntrinsics']),t)
        work_width,work_height=record['workingWidth'],record['workingHeight']
        camera={'ext':np.array(c['worldToCamera']),'fullK':k,
                'workK':np.linalg.inv(np.array(t['workingToModelPixels']))@np.array(c['modelIntrinsics']),
                'workSize':[work_width,work_height]}
        cameras.append(camera)
        points=camera_project(vertices,camera['ext'],camera['workK'])
        z,ids,b=raster(points,faces,work_width,work_height); del ids,b
        mask=np.array(Image.open(a.inputs/record['mask']).convert('L').resize((work_width,work_height),Image.Resampling.LANCZOS))>200
        distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
        distances.append(distance); buffers.append(z)
        valid,md,dz=valid_projection(centers,camera,z,distance,.0025)
        if a.full_mask:
            full_mask=np.array(Image.open(a.inputs/record['fullMask']).convert('L'))
            valid &= sample(full_mask,camera_project(centers,camera['ext'],camera['fullK'])[:,:2])>=254
            del full_mask
        camera_center=-camera['ext'][:,:3].T@camera['ext'][:,3]
        direction=camera_center-centers; length=np.linalg.norm(direction,axis=1)
        facing=np.maximum((normals*direction).sum(1)/length,0)
        angles[:,i]=facing*np.clip(md/3,0,1)*valid
        density[:,i]=(camera['workK'][0,0]/length)**2*valid
        # Favor clear, near-frontal projections and continuous patches, not averaged detail.
        scores[:,i]=(facing**3)*np.clip(md/3,0,1)*valid
        im=np.array(Image.open(a.inputs/record['image']).convert('RGB').resize((work_width,work_height),Image.Resampling.LANCZOS))
        colors.append(sample(im,camera_project(centers,camera['ext'],camera['workK'])[:,:2]).astype('float32'))
        luma=im.astype('float32')@np.array([.2126,.7152,.0722],dtype='float32')
        if a.quality_scale>0:
            luma=cv2.GaussianBlur(luma*mask,(0,0),a.quality_scale)/np.maximum(cv2.GaussianBlur(mask.astype('float32'),(0,0),a.quality_scale),1e-5)
        quality_luma.append(sample(luma,camera_project(centers,camera['ext'],camera['workK'])[:,:2]))
        print('Visibility',record['name'],int(valid.sum()),flush=True)
    if a.selection=='quality':
        # Penalize near-black/clipped observations, not illumination differences
        # between otherwise usable photos. Preferring the brightest source creates
        # patchwork when one view is sunlit and another is in shade.
        luminance=np.stack(quality_luma,axis=1)
        exposure_quality=np.clip(luminance/24,.35,1.)*np.clip((255-luminance)/12,.35,1.)
        relative_density=density/np.maximum(density.max(axis=1)[:,None],1e-6)
        scores=angles*relative_density*exposure_quality
    gains=np.ones((photo_count,3)); overlap=[]
    if a.exposure:
        # Solve robust log-gain differences on jointly visible face centers, anchored to photo 0.
        rows=[]; values=[]
        for i in range(photo_count):
            for j in range(i+1,photo_count):
                use=(scores[:,i]>.15)&(scores[:,j]>.15)&(np.min(colors[i],axis=1)>15)&(np.min(colors[j],axis=1)>15)
                if use.sum()<200: continue
                ratio=np.median(np.log(np.maximum(linear(colors[j][use]),1e-5)/np.maximum(linear(colors[i][use]),1e-5)),axis=0)
                row=np.zeros(photo_count); row[i]=1; row[j]=-1
                rows.append(row); values.append(ratio); overlap.append({'i':i,'j':j,'faces':int(use.sum()),'logDifference':ratio.tolist()})
        if rows:
            anchor=np.zeros(photo_count);anchor[0]=10
            rows.append(anchor); values.append(np.zeros(3))
            logg=np.linalg.lstsq(np.array(rows),np.array(values),rcond=None)[0]
            gains=np.clip(np.exp(logg),.65,1.6)
    labels=np.argmax(scores,axis=1)
    possible=scores.max(axis=1)>1e-6
    adjacency=mesh.face_adjacency
    mat=coo_matrix((np.ones(len(adjacency)*2),(np.r_[adjacency[:,0],adjacency[:,1]],np.r_[adjacency[:,1],adjacency[:,0]])),shape=(len(faces),len(faces))).tocsr()
    for iteration in range(a.smooth):
        votes=mat@np.eye(photo_count,dtype='float32')[labels]
        objective=scores+.10*votes
        objective[scores<=1e-6]=-1
        labels=np.argmax(objective,axis=1)
    labels[~possible]=-1
    preferences=[]
    if a.safe_fill:
        coarse=np.tile([145.,120.,90.],(len(faces),1));known=possible.copy()
        for i in range(photo_count):
            pick=labels==i
            coarse[pick]=encode(linear(colors[i][pick])*gains[i])
        for _ in range(64):
            count=mat@known.astype('float32');new=(~known)&(count>0)
            if not new.any():break
            sums=mat@(coarse*known[:,None]);coarse[new]=sums[new]/count[new,None];known|=new
        fallback=coarse.clip(0,255).astype('uint8')
    np.savez_compressed(a.output/'face-selection.npz',scores=scores,photoIndex=labels,completion=completion)
    uvdata=np.load(a.uv/'uv.npz'); vmap=uvdata['vertexMap']; uvfaces=uvdata['faces']; uv=uvdata['uv']
    width,height=int(uvdata['width']),int(uvdata['height'])
    assert np.array_equal(vmap[uvfaces],faces)
    pixels=np.c_[uv[:,0]*width-.5,(1-uv[:,1])*height-.5,np.ones(len(uv))].astype('float32')
    _,face_ids,bary=raster(pixels,uvfaces,width,height)
    active=face_ids>=0; indices=np.flatnonzero(active)
    flat_ids=face_ids.ravel(); flat_bary=bary.reshape(-1,2)
    texture=np.zeros((height*width,3),dtype='uint8'); sources=np.zeros(height*width,dtype='uint8')
    quality=np.zeros(height*width,dtype='uint8')
    texture[indices]=fallback[flat_ids[indices]]
    for i,record in enumerate(records):
        chosen=indices[labels[flat_ids[indices]]==i]
        im=np.array(Image.open(a.inputs/record['fullImage']).convert('RGB'))
        full_mask=np.array(Image.open(a.inputs/record['fullMask']).convert('L')) if a.full_mask else None
        for start in range(0,len(chosen),150000):
            pixel=chosen[start:start+150000]; fid=flat_ids[pixel]; b=flat_bary[pixel]
            weights=np.c_[b,1-b.sum(1)]
            point=(vertices[faces[fid]]*weights[:,:,None]).sum(1)
            valid,md,dz=valid_projection(point,cameras[i],buffers[i],distances[i],.0025)
            source_xy=camera_project(point,cameras[i]['ext'],cameras[i]['fullK'])[:,:2]
            if a.full_mask:valid &= sample(full_mask,source_xy)>=254
            rgb=encode(linear(sample(im,source_xy))*gains[i]) if a.exposure else sample(im,source_xy)
            selected=pixel[valid]
            texture[selected]=np.clip(rgb[valid],0,255).round().astype('uint8')
            sources[selected]=i+1
            quality[selected]=np.where((md[valid]>=2)&(scores[fid[valid],i]>=.15),2,1)
        print('Baked',record['name'],len(chosen),flush=True)
        del im,full_mask
    texture=texture.reshape(height,width,3)
    # Pad only outside the UV surface; never mark padding as photographic coverage.
    dist,nearest=distance_transform_edt(~active,return_indices=True)
    uv_report=json.loads((a.uv/'report.json').read_text())
    pad=(~active)&(dist<=uv_report['paddingPixels'])
    texture[pad]=texture[nearest[0][pad],nearest[1][pad]]
    Image.fromarray(texture).save(a.output/'basecolor.png')
    Image.fromarray(sources.reshape(height,width)).save(a.output/'texture-source.png')
    Image.fromarray(quality.reshape(height,width)).save(a.output/'texture-quality.png')
    import colorsys
    palette=np.array([[125,125,125]]+[[round(v*255) for v in colorsys.hsv_to_rgb(i/photo_count,.75,.95)] for i in range(photo_count)],dtype='uint8')
    marked=palette[sources.reshape(height,width)]; marked[~active]=0
    Image.fromarray(marked).save(a.output/'texture-source-view.png')
    np.savez_compressed(a.output/'geometry-face-map.npz',originalFaceID=uvdata['originalFaceID'],vertexMap=vmap,faces=uvfaces,uv=uv)
    np.savez_compressed(a.output/'atlas-surface.npz',originalFaceID=face_ids)
    np.savez_compressed(a.output/'geometry.npz',vertices=vertices,faces=faces,normals=mesh.vertex_normals,completion=completion)
    shutil.copy2(a.uv/'uv.npz',a.output/'uv.npz')
    shutil.copy2(run/'fusion/report.json',a.output/'fusion.json')
    write_json(a.output/'cameras.json',{'convention':'OpenCV, integer pixel centers; geometry before worldToDisplay',
        'cameras':[{'name':r['name'],'id':r['id'],'source':'learned camera estimate; not measured','worldToCamera':c['ext'].tolist(),
                    'fullIntrinsics':c['fullK'].tolist(),'workingIntrinsics':c['workK'].tolist(),'workingSize':c['workSize'],
                    'uprightSize':[r['uprightWidth'],r['uprightHeight']]} for r,c in zip(records,cameras)]})
    source_pixels=[int((sources==i+1).sum()) for i in range(photo_count)]
    write_json(a.output/'texture-provenance.json',{'kind':'photographic-texture-on-inferred-geometry','sourcePhotos':records,
        'sourceDisplayColors':[{'code':i,'rgb':color.tolist(),
            'photoID':None if i==0 else records[i-1]['id'],
            'meaning':'appearance fill' if i==0 else records[i-1]['name']} for i,color in enumerate(palette)],
        'sourceMap':'texture-source.png','qualityMap':'texture-quality.png','atlasSurface':'atlas-surface.npz',
        'sourceCodes':{'0':'coarse prior appearance, not a new verified photo sample',**{str(i+1):r['name'] for i,r in enumerate(records)}},
        'qualityCodes':{'0':'fallback or outside UV','1':'photo sample near edge/oblique; low confidence','2':'visible interior photo sample; geometry still inferred'},
        'geometrySourceUnchanged':True,'projectionRule':'atlas pixel center -> barycentric original surface -> recorded camera fullIntrinsics -> bilinear PNG sample',
        'exposureGains':gains.tolist(),'exposureOverlaps':overlap,
        'exposureSpace':'linear sRGB multiplicative gains, clipped .65 to 1.6; not intrinsic albedo recovery',
        'selectionPolicy':a.selection,'fullResolutionMaskRequired':a.full_mask,
        'qualityExposureBlurWorkingPixels':a.quality_scale,
        'regionalSourcePreferences':preferences,
        'fallbackPolicy':'low-frequency valid face colors diffused along mesh adjacency' if a.safe_fill else 'stage1 coarse appearance',
        'limitations':['All geometry and cameras inferred. Photo-supported does not mean calibrated geometric truth.',
                       'Unobserved texels keep coarse prior color and source code zero; no generated details.',
                       'Visibility tested against fixed mesh and foreground mask; masks and surface remain approximate.']})
    stats={'status':'candidate-needs-review','uvWidth':width,'uvHeight':height,'activeTexels':int(active.sum()),
           'photoSupportedTexels':int((sources>0).sum()),'photoCoverageTexelFraction':float((sources>0).sum()/active.sum()),
           'photoSupportedFaceAreaFraction':float(mesh.area_faces[possible].sum()/mesh.area),
           'sourceTexelCounts':source_pixels,'fallbackTexels':int(active.sum()-(sources>0).sum()),
           'smoothingIterations':a.smooth,'exposureCorrection':a.exposure,'surfaceUnchanged':True,**budget.finish()}
    stats.update(selectionPolicy=a.selection,qualityExposureBlurWorkingPixels=a.quality_scale,fullResolutionMask=a.full_mask,safeAppearanceFill=a.safe_fill)
    write_json(a.output/'report.json',stats)
    write_json(a.output/'config.json',{'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'inputManifestSHA256':sha(a.inputs/'dataset.json'),'uvSHA256':sha(a.uv/'uv.npz'),
        'geometrySHA256':sha(run/'fusion/face-provenance.npz'),'sourceCodeSHA256':sha(__file__)})
    print(json.dumps(stats),flush=True)


if __name__=='__main__': main()
