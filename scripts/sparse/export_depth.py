#!/usr/bin/env python3
"""Export learned depth as traceable per-view surfaces, without claiming fusion."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def backproject(depth, intrinsics, extrinsics):
    y, x = np.indices(depth.shape)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    camera = (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]
    return (camera - extrinsics[:, 3]) @ extrinsics[:, :3]


def grid_faces(valid, depth):
    h, w = valid.shape
    ids = np.arange(h * w).reshape(h, w)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    faces = np.concatenate((np.stack((a, c, b), -1).reshape(-1, 3),
                            np.stack((b, c, d), -1).reshape(-1, 3)))
    keep = valid.ravel()[faces].all(axis=1)
    z = depth.ravel()[faces]
    keep &= np.ptp(z, axis=1) < .04 * np.median(depth[valid])
    return faces[keep]


def write_viewer(path, vertices, colors, view_ids, names, completion=None):
    # Standalone local point viewer, intentionally independent of the learned backend.
    indices = np.linspace(0, len(vertices) - 1, min(len(vertices), 45000), dtype=int)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    fitted = (vertices[indices] - (low + high) / 2) * (2 / max(high - low))
    data = {"p": fitted.round(5).tolist(), "c": colors[indices, :3].tolist(),
            "v": view_ids[indices].tolist(), "names": names,
            "fused": completion is not None,
            "r": (np.zeros(len(indices), dtype=int) if completion is None else completion[indices].astype(int)).tolist()}
    template = Path(__file__).with_name("point-viewer.html").read_text()
    path.write_text(template.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--quantile", type=float, default=.1, help="Per-view foreground confidence quantile to discard")
    a = p.parse_args()
    data = np.load(a.run / "predictions.npz")
    names = json.loads((a.run / "config.json").read_text())["photoNames"]
    out = a.run / "geometry"
    out.mkdir(exist_ok=False)
    depth, masks = data["depth"], data["masks"]
    points = np.stack([backproject(d, k, e) for d, k, e in zip(depth, data["intrinsics"], data["extrinsics"])])
    all_valid = masks & np.isfinite(points).all(-1) & (depth > 0)
    center = np.median(points[all_valid], axis=0)
    extent = np.percentile(points[all_valid], 99, axis=0) - np.percentile(points[all_valid], 1, axis=0)
    scale = 2 / max(extent)
    world_to_display = np.diag([scale, -scale, -scale, 1.])
    world_to_display[:3, 3] = -world_to_display[:3, :3] @ center
    points_display = (points - center) * [scale, -scale, -scale]
    scene = trimesh.Scene()
    regions = trimesh.Scene()
    cloud, colors, view_ids, groups = [], [], [], []
    size = depth.shape[1]
    sheet = Image.new("RGB", (len(names) * size, size * 2 + 32), "#222222")
    for i, name in enumerate(names):
        valid = all_valid[i]
        confidence = data["confidence"][i]
        threshold = float(np.quantile(confidence[valid], a.quantile))
        valid = valid & (confidence >= threshold)
        faces = grid_faces(valid, depth[i])
        mesh = trimesh.Trimesh(vertices=points_display[i].reshape(-1, 3), faces=faces,
                              vertex_colors=data["rgb"][i].reshape(-1, 3), process=False)
        mesh.remove_unreferenced_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.metadata.update(source="learned-inference", photo=name, crossViewVerified=False)
        scene.add_geometry(mesh, node_name=f"learned-view-{i}", geom_name=f"learned-view-{i}")
        classified = mesh.copy()
        classified.visual.vertex_colors = np.tile([244, 160, 42, 255], (len(mesh.vertices), 1))
        regions.add_geometry(classified, node_name=f"learned-view-{i}", geom_name=f"learned-view-{i}")
        mesh.export(out / f"view-{i}.ply")
        cloud.append(points_display[i][valid])
        colors.append(data["rgb"][i][valid])
        view_ids.append(np.full(valid.sum(), i))
        groups.append({"name": name, "node": f"learned-view-{i}", "source": "learned-inference",
                       "vertices": len(mesh.vertices), "triangles": len(mesh.faces), "points": int(valid.sum()),
                       "confidenceThreshold": threshold, "independentMultiViewValidation": False})
        sheet.paste(Image.fromarray(data["rgb"][i]), (i * size, 32))
        lo, hi = np.percentile(depth[i][all_valid[i]], [2, 98])
        shading = np.clip((depth[i] - lo) / max(hi - lo, 1e-8), 0, 1)
        heat = np.stack((255 * (1 - shading), 100 + 100 * shading, 255 * shading), axis=-1).astype("uint8")
        heat[~masks[i]] = 25
        sheet.paste(Image.fromarray(heat), (i * size, 32 + size))
        ImageDraw.Draw(sheet).text((i * size + 5, 8), name, fill="white")
    cloud, colors, view_ids = np.concatenate(cloud), np.concatenate(colors), np.concatenate(view_ids)
    trimesh.PointCloud(cloud, colors=colors).export(out / "foreground.ply")
    scene.export(out / "per-view-surfaces.glb")
    regions.export(out / "inference-regions.glb")
    np.savez_compressed(out / "point-provenance.npz", position=cloud, rgb=colors, viewIndex=view_ids)
    sheet.save(out / "depth-review.jpg")
    write_viewer(out / "point-viewer.html", cloud, colors, view_ids, names)
    report = {"status": "unvalidated-local-surfaces", "groups": groups,
              "worldToDisplay": world_to_display.tolist(), "confidenceDiscardQuantile": a.quantile,
              "classification": {"learned-inference": {"displayColor": "#f4a02a", "fraction": 1.0}},
              "limitations": ["All geometry and cameras are predictions, not measured depth.",
                              "Per-view surfaces can overlap or disagree; no global fusion or acceptance is implied.",
                              "Colors are sampled from real photos; colored surfaces are still inferred."]}
    (out / "geometry-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    restored = trimesh.load(out / "per-view-surfaces.glb")
    assert sum(len(m.faces) for m in restored.geometry.values()) > 0
    assert np.isfinite(restored.bounds).all()
    print(json.dumps({"vertices": len(cloud), "groups": groups, "output": str(out)}))


if __name__ == "__main__":
    main()
