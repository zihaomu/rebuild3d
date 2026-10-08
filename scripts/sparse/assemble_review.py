#!/usr/bin/env python3
"""Combine original photos, USDZ renders and mask differences; preserve quantitative review."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("fusion", type=Path)
    a = p.parse_args()
    data = np.load(a.run / "predictions.npz")
    names = json.loads((a.run / "config.json").read_text())["photoNames"]
    render = a.fusion / "renders"
    size = data["depth"].shape[1]
    sheet = Image.new("RGB", (size * 4, (size + 32) * len(names)), "#252930")
    rows = []
    for i, name in enumerate(names):
        observed = data["masks"][i]
        actual = np.asarray(Image.open(render / f"view-{i}-mask.png").convert("L")) > 128
        intersection = int((observed & actual).sum())
        union = int((observed | actual).sum())
        rows.append({"name": name, "silhouetteIoU": intersection / union,
                     "observedPixels": int(observed.sum()), "renderedPixels": int(actual.sum()),
                     "extraPixels": int((actual & ~observed).sum()), "missingPixels": int((observed & ~actual).sum())})
        difference = np.full((size, size, 3), 28, dtype="uint8")
        difference[actual & observed] = [65, 180, 105]
        difference[actual & ~observed] = [240, 90, 60]
        difference[~actual & observed] = [65, 130, 250]
        y = i * (size + 32)
        ImageDraw.Draw(sheet).text((8, y + 10), f"{name} / original | same USDZ, photo color | same USDZ, no texture | silhouette difference / IoU {intersection/union:.3f}", fill="white")
        images = [Image.fromarray(data["rgb"][i]), Image.open(render / f"view-{i}-color.png"),
                  Image.open(render / f"view-{i}-clay.png"), Image.fromarray(difference)]
        for j, image in enumerate(images):
            sheet.paste(image, (j * size, y + 32))
    sheet.save(a.fusion / "seven-view-review.jpg", quality=95)
    frames = [Image.open(path).convert("RGB") for path in sorted(render.glob("turntable-*.png"))]
    frames[0].save(a.fusion / "untextured-turntable.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
    contact = Image.new("RGB", (640 * 3, 640 * 2), "#202020")
    for k, i in enumerate((0, 6, 12, 18, 24, 30)):
        contact.paste(frames[i], ((k % 3) * 640, (k // 3) * 640))
    contact.save(a.fusion / "untextured-six-views.jpg", quality=95)
    report = {"views": rows, "meanSilhouetteIoU": float(np.mean([r["silhouetteIoU"] for r in rows])),
              "method": "Same exported USDZ rendered by SceneKit at all seven predicted cameras, 518x518, 4x MSAA mask threshold 128.",
              "limitations": "Input-view consistency only; camera predictions and rough hand-edited masks are not ground truth. Texture does not establish geometric accuracy."}
    (a.fusion / "review-metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
