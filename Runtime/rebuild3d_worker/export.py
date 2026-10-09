"""Export actual per-pixel UV textures to GLB and self-contained USDZ."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
from .common import sha, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--name',default='model');p.add_argument('--texture',default='basecolor.png')
    a=p.parse_args();root=a.directory;name=a.name
    assert Path(name).name==name and Path(a.texture).name==a.texture
    assert not (root/f'{name}.glb').exists() and not (root/f'{name}.usdz').exists()
    d=np.load(root/'geometry.npz'); u=np.load(root/'uv.npz')
    display=np.array(json.loads((root/'fusion.json').read_text())['worldToDisplay'])
    vertices=d['vertices'][u['vertexMap']]@display[:3,:3].T+display[:3,3]
    normals=d['normals'][u['vertexMap']]@display[:3,:3].T
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    uv=u['uv']; faces=u['faces']; completed=d['completion']
    texture=Image.open(root/a.texture).convert('RGB')
    mat=trimesh.visual.material.PBRMaterial(name='PhotographicBaseColor',baseColorTexture=texture,
        metallicFactor=0.,roughnessFactor=.8,doubleSided=True)
    scene=trimesh.Scene()
    stage=Usd.Stage.CreateNew(str(root/f'{name}.usdc'))
    x=UsdGeom.Xform.Define(stage,'/Object');stage.SetDefaultPrim(x.GetPrim());UsdGeom.SetStageUpAxis(stage,'Y')
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    x.GetPrim().SetCustomDataByKey('rebuild3d:geometry','All geometry inferred; relative scale; see companion provenance')
    material=UsdShade.Material.Define(stage,'/Object/PhotoMaterial')
    surface=UsdShade.Shader.Define(stage,'/Object/PhotoMaterial/Surface');surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8);surface.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    reader=UsdShade.Shader.Define(stage,'/Object/PhotoMaterial/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    tex=UsdShade.Shader.Define(stage,'/Object/PhotoMaterial/Texture');tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(a.texture))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    surface.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    surface.CreateOutput('surface',Sdf.ValueTypeNames.Token);material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),'surface')
    for i,(label,select) in enumerate([('learned-depth',~completed),('silhouette-completion',completed)]):
        if not select.any(): continue
        m=trimesh.Trimesh(vertices=vertices,faces=faces[select],vertex_normals=normals,process=False)
        m.visual=trimesh.visual.TextureVisuals(uv=uv,material=mat)
        m.remove_unreferenced_vertices()
        scene.add_geometry(m,node_name=label,geom_name=label)
        mesh=UsdGeom.Mesh.Define(stage,f'/Object/region_{i}');mesh.GetPrim().SetDisplayName(label)
        mesh.GetPrim().SetCustomDataByKey('rebuild3d:sourceRegion',label)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertices,dtype='float32')))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces),3,dtype='int32')))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m.faces,dtype='int32').ravel()))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertex_normals,dtype='float32')))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex);mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(m.visual.uv,dtype='float32')))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    scene.export(root/f'{name}.glb');stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(root/f'{name}.usdc')),str(root/f'{name}.usdz'))
    restored=trimesh.load(root/f'{name}.glb',force='scene')
    assert sum(len(m.faces) for m in restored.geometry.values())==len(faces)
    assert all(m.visual.kind=='texture' for m in restored.geometry.values())
    report_name='asset-report.json' if name=='model' else f'{name}-asset-report.json'
    write_json(root/report_name,{'triangles':len(faces),'textureSize':list(texture.size),'textureFile':a.texture,
        'glbSHA256':sha(root/f'{name}.glb'),'usdzSHA256':sha(root/f'{name}.usdz'),'basecolorSHA256':sha(root/a.texture),
        'allGeometryInferred':True,'perPixelTexture':True,'glbBytes':(root/f'{name}.glb').stat().st_size,'usdzBytes':(root/f'{name}.usdz').stat().st_size})
    print((root/report_name).read_text())


if __name__=='__main__':main()
