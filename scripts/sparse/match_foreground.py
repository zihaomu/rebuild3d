#!/usr/bin/env python3
"""Save inspectable foreground-only SIFT correspondences for all 21 photo pairs."""
import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cv2.setRNGSeed(0)
    cv2.setNumThreads(4)
    records = json.loads((a.dataset / "dataset.json").read_text())["records"]
    sift = cv2.SIFT_create(nfeatures=16000)
    photos, points, descriptors = [], [], []
    features = {}
    for i, r in enumerate(records):
        image = cv2.imread(str(a.dataset / r["image"]))
        mask = cv2.imread(str(a.dataset / r["mask"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.erode((mask >= 128).astype("uint8") * 255, np.ones((7, 7), "uint8"))
        keypoints, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
        xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        points.append(xy)
        descriptors.append(desc)
        photos.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        features[f"points_{i}"] = xy
        features[f"descriptors_{i}"] = desc
    np.savez_compressed(a.output / "features.npz", **features)
    matcher = cv2.BFMatcher()
    pairs = []
    for i, j in itertools.combinations(range(len(records)), 2):
        forward = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(descriptors[i], descriptors[j], k=2)
                   if m.distance < .75 * n.distance}
        backward = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(descriptors[j], descriptors[i], k=2)
                    if m.distance < .75 * n.distance}
        matches = np.array([(s, t) for s, t in forward.items() if backward.get(t) == s], dtype=int).reshape(-1, 2)
        pi, pj = points[i][matches[:, 0]], points[j][matches[:, 1]]
        F, keep = (None, None)
        if len(matches) >= 8:
            F, keep = cv2.findFundamentalMat(pi, pj, cv2.FM_RANSAC, 2., .999, 10000)
        keep = np.zeros(len(matches), dtype=bool) if keep is None else keep.ravel().astype(bool)
        row = {"indices": [i, j], "names": [records[i]["name"], records[j]["name"]],
               "mutualRatioMatches": len(matches), "fundamentalInliers": int(keep.sum()),
               "fundamental": None if F is None else F.tolist(),
               "inlierFeatureIndices": matches[keep].tolist(),
               "pointsA": pi[keep].tolist(), "pointsB": pj[keep].tolist(),
               "warning": "RANSAC inliers are candidate correspondences, not verified physical matches or camera registration"}
        pairs.append(row)
        panel = Image.new("RGB", (1200, 830), "#222222")
        panel.paste(photos[i].resize((600, 800)), (0, 30))
        panel.paste(photos[j].resize((600, 800)), (600, 30))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), f"{row['names']} / mutual {len(matches)} / F inliers {keep.sum()}", fill="white")
        for k, (xy1, xy2) in enumerate(zip(pi[keep][:60], pj[keep][:60])):
            color = tuple(int(v) for v in np.random.default_rng(k).integers(80, 256, 3))
            x1, y1 = xy1 / 2 + [0, 30]
            x2, y2 = xy2 / 2 + [600, 30]
            draw.line((x1, y1, x2, y2), fill=color, width=1)
            for x, y in [(x1, y1), (x2, y2)]:
                draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        panel.save(a.output / f"pair-{i}-{j}.jpg")
        print(i, j, len(matches), int(keep.sum()), flush=True)
    report = {"method": "SIFT 16000, foreground eroded 3px; mutual ratio .75; F RANSAC 2px, .999, 10000, seed 0",
              "featureCounts": [len(p) for p in points], "pairs": pairs,
              "elapsedSeconds": time.monotonic() - start, "opencv": cv2.__version__}
    (a.output / "matches.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
