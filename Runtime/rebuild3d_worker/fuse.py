#!/usr/bin/env python3
"""Fuse predicted depths with photographed silhouette constraints and explicit completion labels."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from .geometry import backproject, project


def sample(image, xy, order=1, outside=0):
    if image.ndim == 3:
        return np.stack([sample(image[..., i], xy, order, outside) for i in range(image.shape[2])], -1)
    return map_coordinates(image, [xy[:, 1], xy[:, 0]], order=order, mode="constant", cval=outside, prefilter=False)


def retain_component(index, distance, spacing, area, main_area, main_extent, face_support):
    """Preserve observed detached parts without inventing a bridge to the main mesh."""
    total = max(float(area.sum()), 1e-12)
    supported = float(area[face_support.any(axis=1)].sum() / total)
    view_areas = (area[:, None] * face_support).sum(axis=0) / total
    # A broken thin connector can put a real accessory beyond five voxels. Require
    # substantial size and independent views, rather than relaxing the dust rule.
    observed_part = (total >= main_area * .01 and distance <= main_extent * .1
                     and supported >= .9 and np.count_nonzero(view_areas >= .1) >= 2)
    keep = index == 0 or (distance <= spacing * 5 and supported >= .6) or observed_part
    reason = ("main" if index == 0 else "nearby-depth-supported" if distance <= spacing * 5 and supported >= .6
              else "substantial-multiview-part" if observed_part else "unsupported-or-detached-dust")
    return bool(keep), reason, supported, view_areas.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--silhouette-tolerance", type=float, default=2.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    data = np.load(a.run / "predictions.npz")
    transforms = json.loads((a.run / "pixel-transforms.json").read_text())
    names = [t["name"] for t in transforms]
    depth, masks, confidence = data["depth"], data["masks"], data["confidence"]
    ext, intr = data["extrinsics"], data["intrinsics"]
    clouds = [backproject(d, k, e)[m] for d, k, e, m in zip(depth, intr, ext, masks)]
    cloud = np.concatenate(clouds)
    votes = np.zeros(len(cloud), dtype="uint8")
    for i in range(len(names)):
        xy, z = project(cloud, ext[i], intr[i])
        expanded = cv2.dilate(masks[i].astype("uint8"), np.ones((11, 11), "uint8"))
        votes += ((sample(expanded, xy, order=0) > 0) & (z > 0)).astype("uint8")
    bounds_votes = max(2, int(np.ceil(len(names) * 0.7)))
    interior_votes = max(2, int(np.ceil(len(names) * 0.55)))
    bounded = cloud[votes >= bounds_votes]
    assert len(bounded) > 1000, "Insufficient shared foreground for a unified volume"
    lo, hi = np.percentile(bounded, [0, 100], axis=0)
    spacing = float(max(hi - lo) / (a.resolution - 12))
    lo, hi = lo - 6 * spacing, hi + 6 * spacing
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    axes = [lo[i] + np.arange(dims[i]) * spacing for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype("float32")
    truncation = spacing * 4
    print("Volume", dims.tolist(), "spacing", spacing, "voxels", len(points), flush=True)
    sdf_sum = np.zeros(len(points), dtype="float32")
    weight = np.zeros(len(points), dtype="float32")
    foreground_views = np.zeros(len(points), dtype="uint8")
    hull = np.full(len(points), -truncation, dtype="float32")
    silhouette_fields = []
    for i in range(len(names)):
        binary = masks[i].astype("uint8")
        signed = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5) - cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        silhouette_fields.append(signed)
        for offset in range(0, len(points), 400000):
            sl = slice(offset, offset + 400000)
            xy, z = project(points[sl], ext[i], intr[i])
            # Padding and areas outside the physical photo are not observations.
            left, top = transforms[i]["paddingLeftTop"]
            width, height = transforms[i]["resizedSize"]
            seen = (z > 0) & (xy[:, 0] >= left) & (xy[:, 0] < left + width - 1) & (xy[:, 1] >= top) & (xy[:, 1] < top + height - 1)
            distance = (sample(signed, xy, outside=1e3) - a.silhouette_tolerance) * z / ((intr[i, 0, 0] + intr[i, 1, 1]) / 2)
            hull[sl] = np.maximum(hull[sl], np.where(seen, distance, -truncation))
            observed = sample(depth[i], xy)
            difference = observed - z
            inside = sample(binary, xy, order=0) > 0
            foreground_views[sl] += (seen & inside).astype("uint8")
            use = seen & inside & (difference >= -truncation)
            certainty = np.clip(sample(confidence[i], xy) - 1, .1, 5.) * use
            sdf_sum[sl] += np.clip(difference, -truncation, truncation) * certainty
            weight[sl] += certainty
        print("Integrated", names[i], flush=True)
    # Unobserved interior is a silhouette-based solid completion, not measured depth.
    # Voxels outside most physical photo frusta are unknown exterior, not a solid.
    interior = np.where(foreground_views >= interior_votes, -truncation, truncation)
    field = np.where(weight > 0, sdf_sum / np.maximum(weight, 1e-6), interior)
    field = np.maximum(field, hull).reshape(tuple(dims))
    field[[0, -1], :, :] = truncation
    field[:, [0, -1], :] = truncation
    field[:, :, [0, -1]] = truncation
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(spacing,) * 3, gradient_direction="ascent")
    vertices += lo
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    parts = mesh.split(only_watertight=False)
    # Keep thin detached instrument/cloth components if substantive; discard voxel dust.
    substantial = [m for m in parts if len(m.faces) >= 80]
    removed_faces = sum(len(m.faces) for m in parts if len(m.faces) < 80)
    mesh = trimesh.util.concatenate(substantial)
    trimesh.smoothing.filter_taubin(mesh, lamb=.5, nu=.53, iterations=3)
    mesh.fix_normals(multibody=True)
    vertices = mesh.vertices.copy()
    normals = mesh.vertex_normals.copy()
    support = np.zeros((len(vertices), len(names)), dtype=bool)
    best = np.full(len(vertices), -1.)
    colors = np.tile([145, 120, 90], (len(vertices), 1)).astype("uint8")
    color_view = np.full(len(vertices), -1, dtype=int)
    for i in range(len(names)):
        xy, z = project(vertices, ext[i], intr[i])
        observed = sample(depth[i], xy)
        inside = sample(masks[i].astype("uint8"), xy, order=0) > 0
        delta = abs(z - observed)
        support[:, i] = inside & (z > 0) & (delta < truncation * 1.5)
        center = -ext[i, :, :3].T @ ext[i, :, 3]
        direction = center - vertices
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-8)
        facing = np.maximum(np.sum(normals * direction, axis=1), 0)
        score = facing * np.exp(-delta / truncation) * support[:, i]
        change = (score > best) & support[:, i]
        best[change] = score[change]
        colors[change] = sample(data["rgb"][i].astype("float32"), xy[change]).clip(0, 255).astype("uint8")
        color_view[change] = i
    # A face requires common depth support across every vertex to stay inferred-depth.
    face_support = support[mesh.faces].all(axis=1)
    completion = ~face_support.any(axis=1)
    # Keep the main object and nearby detached details that have substantial depth
    # support. Silhouette-only slabs disconnected from it are not justified parts.
    components = sorted(trimesh.graph.connected_components(mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)), min_len=1), key=len, reverse=True)
    tree = cKDTree(vertices[np.unique(mesh.faces[components[0]])])
    main_area = float(mesh.area_faces[components[0]].sum())
    main_extent = float(np.ptp(vertices[np.unique(mesh.faces[components[0]])], axis=0).max())
    retained, component_review = [], []
    for index, component in enumerate(components):
        component_vertices = np.unique(mesh.faces[component])
        distance = float(tree.query(vertices[component_vertices])[0].min())
        area = mesh.area_faces[component]
        keep, reason, supported_fraction, view_areas = retain_component(
            index, distance, spacing, area, main_area, main_extent, face_support[component])
        component_review.append({"triangles": len(component), "distanceToMain": distance,
            "depthSupportedAreaFraction": supported_fraction, "retained": keep,
            "retentionReason": reason, "areaRelativeToMain": float(area.sum() / max(main_area, 1e-12)),
            "supportedAreaFractionPerView": view_areas})
        if keep:
            retained.extend(component)
    retained = np.sort(np.array(retained, dtype=int))
    removed_disconnected = len(mesh.faces) - len(retained)
    kept_vertices, remap = np.unique(mesh.faces[retained], return_inverse=True)
    vertices, colors, color_view = vertices[kept_vertices], colors[kept_vertices], color_view[kept_vertices]
    face_support, completion = face_support[retained], completion[retained]
    mesh = trimesh.Trimesh(vertices=vertices, faces=remap.reshape(-1, 3), process=False)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    scale = 2 / max(mesh.extents)
    matrix = np.diag([scale, -scale, -scale, 1.])
    matrix[:3, 3] = -matrix[:3, :3] @ center
    display = matrix.tolist()
    mesh.visual.vertex_colors = colors
    mesh.apply_transform(np.array(display))
    mesh.export(a.output / "model.ply")
    scene, source_scene = trimesh.Scene(), trimesh.Scene()
    for label, selected, color in (("learned-depth", ~completion, [244, 160, 42, 255]),
                                   ("silhouette-completion", completion, [197, 88, 235, 255])):
        if not selected.any():
            continue
        part = mesh.submesh([np.flatnonzero(selected)], append=True, repair=False)
        part.metadata.update(source=label, measured=False)
        scene.add_geometry(part, node_name=label, geom_name=label)
        marked = part.copy()
        marked.visual.vertex_colors = np.tile(color, (len(marked.vertices), 1))
        source_scene.add_geometry(marked, node_name=label, geom_name=label)
    scene.export(a.output / "model.glb")
    source_scene.export(a.output / "provenance.glb")
    learned = mesh.submesh([np.flatnonzero(~completion)], append=True, repair=False)
    learned.export(a.output / "without-completion.glb")
    np.savez_compressed(a.output / "face-provenance.npz", completion=completion,
                        supportingPhotoIndices=face_support, vertexColorPhotoIndex=color_view,
                        verticesBeforeDisplay=vertices, faces=mesh.faces)
    vertex_completion = np.zeros(len(mesh.vertices), dtype=bool)
    vertex_completion[np.unique(mesh.faces[completion])] = True
    report = {"status": "candidate-needs-render-review", "inputRun": str(a.run), "photos": names,
              "resolution": a.resolution, "gridDimensions": dims.tolist(), "voxelSpacingWorld": spacing,
              "boundsRule": {"boundsVotes": bounds_votes, "interiorVotes": interior_votes, "photos": len(names)},
              "boundedPointCount": len(bounded),
              "truncationWorld": truncation, "silhouetteTolerancePixels": a.silhouette_tolerance,
              "worldToDisplay": display, "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
              "watertight": bool(mesh.is_watertight), "bounds": mesh.bounds.tolist(),
              "removedDustTriangles": removed_faces, "removedDisconnectedTriangles": removed_disconnected,
              "componentReview": component_review,
              "componentRule": "Main; nearby parts within five voxels with 60% depth support; or substantial parts (1% main area, within 10% main extent) with 90% support and at least two views each supporting 10% area. No artificial connectors.",
              "componentsKept": sum(c["retained"] for c in component_review),
              "completionTriangles": int(completion.sum()), "completionFaceFraction": float(completion.mean()),
              "learnedDepthTriangles": int((~completion).sum()), "elapsedSeconds": time.monotonic() - start,
              "provenance": {"learned-depth": "Orange: learned geometry constrained by original-photo silhouettes, no measured depth.",
                             "silhouette-completion": "Purple: shell/closure not supported near predicted depth in any view, inferred from silhouette volume."},
              "limitations": ["Relative scale only; all geometry and cameras are inferred.",
                              "Support IDs refer to agreement with predicted depths, not independent depth measurements.",
                              "Smoothing and solid interior are approximations; before-completion model retained."]}
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
