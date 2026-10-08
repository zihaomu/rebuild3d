#!/usr/bin/env python3
"""Export named provenance regions and photographed vertex colors to USDZ."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt


def export(source, output):
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    scene = trimesh.load(source, force="scene")
    stage_path = output.with_suffix(".usdc")
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Statue")
    root.GetPrim().SetCustomDataByKey("rebuild3d:scale", "relative normalized units; no physical size calibration")
    root.GetPrim().SetCustomDataByKey("rebuild3d:geometry", "approximate learned reconstruction; see sidecar provenance")
    stage.SetDefaultPrim(root.GetPrim())
    counts = []
    for i, node in enumerate(scene.graph.nodes_geometry):
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(transform)
        material = UsdShade.Material.Define(stage, f"/Statue/PhotoColor_{i}")
        shader = UsdShade.Shader.Define(stage, f"/Statue/PhotoColor_{i}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.8)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
        # Bake a constant color per tiny triangle. A real UV texture is supported
        # by Apple viewers that do not resolve a float3 displayColor reader.
        face_colors = np.asarray(mesh.visual.vertex_colors[:, :3])[mesh.faces].mean(axis=1).round().astype("uint8")
        tile = 2
        side = 2 ** int(np.ceil(np.log2(np.ceil(np.sqrt(len(face_colors))) * tile)))
        columns = side // tile
        index = np.arange(len(face_colors))
        x, y = (index % columns) * tile, (index // columns) * tile
        pixels = np.zeros((side, side, 3), dtype="uint8")
        for dx in range(tile):
            for dy in range(tile):
                pixels[y + dy, x + dx] = face_colors
        texture_path = output.with_name(output.stem + f"-region-{i}.png")
        Image.fromarray(pixels).save(texture_path)
        uv = np.c_[(x + tile / 2) / side, 1 - (y + tile / 2) / side].astype("float32")
        reader = UsdShade.Shader.Define(stage, f"/Statue/PhotoColor_{i}/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, f"/Statue/PhotoColor_{i}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path.name))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        prim = UsdGeom.Mesh.Define(stage, f"/Statue/region_{i}")
        prim.GetPrim().SetDisplayName(str(node))
        prim.GetPrim().SetCustomDataByKey("rebuild3d:sourceRegion", str(node))
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype="float32")))
        prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype="int32")))
        prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype="int32").ravel()))
        prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        prim.CreateDoubleSidedAttr(True)
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype="float32")))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
            Vt.Vec2fArray.FromNumpy(np.repeat(uv, 3, axis=0)))
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
        counts.append({"node": str(node), "vertices": len(mesh.vertices), "triangles": len(mesh.faces)})
    stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(stage_path)), str(output))
    restored = Usd.Stage.Open(str(output))
    assert restored and sum(p.IsA(UsdGeom.Mesh) for p in restored.Traverse()) == len(counts)
    return {"file": str(output), "bytes": output.stat().st_size, "regions": counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, help="New directory for an alternative export")
    a = p.parse_args()
    output = a.output or a.directory
    if a.output:
        output.mkdir(parents=True, exist_ok=False)
    reports = [export(a.directory / f"{name}.glb", output / f"{name}.usdz") for name in ("model", "provenance")]
    (output / "usdz-export.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
