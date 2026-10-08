#!/usr/bin/env python3
"""Check learned cameras against real feature matches and leave-one-view-out silhouettes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from export_depth import backproject


def project(points, extrinsics, intrinsics):
    camera = points @ extrinsics[:, :3].T + extrinsics[:, 3]
    pixels = camera @ intrinsics.T
    return pixels[..., :2] / np.maximum(pixels[..., 2:], 1e-10), camera[..., 2]


def skew(t):
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def splat(points, rgb, extrinsics, intrinsics, size):
    pixels, z = project(points, extrinsics, intrinsics)
    uv = np.rint(pixels).astype(int)
    valid = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
    uv, z, rgb = uv[valid], z[valid], rgb[valid]
    order = np.argsort(z)
    ids = uv[order, 1] * size + uv[order, 0]
    ids, first = np.unique(ids, return_index=True)
    canvas = np.full((size * size, 3), 28, dtype="uint8")
    canvas[ids] = rgb[order[first]]
    mask = np.zeros(size * size, dtype="uint8")
    mask[ids] = 1
    mask = cv2.morphologyEx(mask.reshape(size, size), cv2.MORPH_CLOSE, np.ones((3, 3), "uint8")) > 0
    return canvas.reshape(size, size, 3), mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("matches", type=Path)
    a = p.parse_args()
    out = a.run / "evaluation"
    out.mkdir(exist_ok=False)
    data = np.load(a.run / "predictions.npz")
    transforms = json.loads((a.run / "pixel-transforms.json").read_text())
    names = [t["name"] for t in transforms]
    depth, masks = data["depth"], data["masks"]
    ext, intr = data["extrinsics"], data["intrinsics"]
    world = np.stack([backproject(d, k, e) for d, k, e in zip(depth, intr, ext)])
    size = depth.shape[1]
    cloud = np.concatenate([w[m] for w, m in zip(world, masks)])
    color = np.concatenate([r[m] for r, m in zip(data["rgb"], masks)])
    ids = np.concatenate([np.full(m.sum(), i) for i, m in enumerate(masks)])
    rows = []
    sheet = Image.new("RGB", (size * 3, (size + 30) * len(names)), "#20252c")
    for i, name in enumerate(names):
        rendered, all_mask = splat(cloud, color, ext[i], intr[i], size)
        _, loo = splat(cloud[ids != i], color[ids != i], ext[i], intr[i], size)
        target = masks[i]
        iou = lambda x: float((x & target).sum() / max((x | target).sum(), 1))
        rows.append({"name": name, "allSurfaceSilhouetteIoU": iou(all_mask),
                     "leaveOwnSurfaceOutSilhouetteIoU": iou(loo)})
        diff = np.full((size, size, 3), 28, dtype="uint8")
        diff[loo & target] = [65, 180, 105]
        diff[loo & ~target] = [240, 90, 60]
        diff[~loo & target] = [65, 130, 250]
        y = i * (size + 30)
        ImageDraw.Draw(sheet).text((5, y + 6), f"{name}  original | all surfaces | leave-own-out: green agree, red extra, blue missing; IoU {iou(loo):.3f}", fill="white")
        for j, image in enumerate((data["rgb"][i], rendered, diff)):
            sheet.paste(Image.fromarray(image), (j * size, y + 30))
    sheet.save(out / "seven-view-comparison.jpg")
    pairs = []
    for row in json.loads(a.matches.read_text())["pairs"]:
        if not all(n in names for n in row["names"]) or not row["pointsA"]:
            continue
        i, j = [names.index(n) for n in row["names"]]
        left, right = [] , []
        for key, index, dest in (("pointsA", i, left), ("pointsB", j, right)):
            xy = np.array(row[key])
            xy = np.c_[xy, np.ones(len(xy))] @ np.array(transforms[index]["workingToModelPixels"]).T
            dest.extend(xy)
        left, right = np.array(left), np.array(right)
        relative_r = ext[j, :, :3] @ ext[i, :, :3].T
        relative_t = ext[j, :, 3] - relative_r @ ext[i, :, 3]
        f = np.linalg.inv(intr[j]).T @ skew(relative_t) @ relative_r @ np.linalg.inv(intr[i])
        fl, fr = left @ f.T, right @ f
        residual = np.abs(np.sum(right * fl, axis=1)) / np.sqrt((fl[:, :2]**2).sum(1) + (fr[:, :2]**2).sum(1) + 1e-20)
        uv = np.rint(left[:, :2]).astype(int)
        projected, z = project(world[i, uv[:, 1], uv[:, 0]], ext[j], intr[j])
        error = np.linalg.norm(projected - right[:, :2], axis=1)
        pairs.append({"names": row["names"], "candidateMatches": len(left),
                      "medianSampsonPixels": float(np.median(residual)),
                      "medianDepthReprojectionPixels": float(np.median(error)),
                      "medianDepthReprojectionImageDiagonalFraction": float(np.median(error) / (size * np.sqrt(2))),
                      "positiveDepthFraction": float(np.mean(z > 0))})
    centers = np.stack([-e[:, :3].T @ e[:, 3] for e in ext])
    report = {"views": rows, "featurePairs": pairs, "cameraCenters": centers.tolist(),
              "meanLeaveOwnOutIoU": float(np.mean([r["leaveOwnSurfaceOutSilhouetteIoU"] for r in rows])),
              "method": "Unfiltered predicted foreground points, nearest-pixel z-buffer, 3x3 closing for silhouettes; each leave-own-out view excludes its own predicted surface.",
              "limitations": "These are input consistency diagnostics, not scan accuracy. Feature matches are RANSAC candidates, not all manually verified; silhouette cameras are learned estimates."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
