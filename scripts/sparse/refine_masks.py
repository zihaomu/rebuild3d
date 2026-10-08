#!/usr/bin/env python3
"""Apply recorded object/plinth boundaries without changing original photographs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path(__file__).with_name("statue-mask-boundaries.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output must be a new directory")
    annotations = json.loads(args.annotations.read_text())
    dataset = json.loads((args.input / "dataset.json").read_text())
    shutil.copytree(args.input, args.output)
    shutil.copy2(args.annotations, args.output / "manual-mask-annotations.json")
    (args.output / "comparisons").mkdir()
    sheet = Image.new("RGB", (7 * 300, 832), "#222222")
    for i, record in enumerate(dataset["records"]):
        width, height = record["workingWidth"], record["workingHeight"]
        assert (width, height) == (1200, 1600)
        full_path = args.output / record["fullMask"]
        mask = Image.open(full_path).convert("L")
        fw, fh = mask.size
        polygon = [(0, 0), (width, 0), *reversed(annotations["lowerBoundaries"][record["name"]])]
        polygon = [((x + .5) * fw / width - .5, (y + .5) * fh / height - .5) for x, y in polygon]
        keep = Image.new("L", mask.size)
        ImageDraw.Draw(keep).polygon(polygon, fill=255)
        mask = Image.fromarray(np.minimum(np.asarray(mask), np.asarray(keep)))
        mask.save(full_path)
        working = mask.resize((width, height), Image.Resampling.LANCZOS)
        working.save(args.output / record["mask"])
        record["maskRevision"] = 2
        record["maskMethod"] += "; manual lower rock boundary (annotations retained)"
        record["maskSHA256"] = digest(args.output / record["mask"])
        record["fullMaskSHA256"] = digest(full_path)
        record["reviewStatus"] = "visually-reviewed; coarse boundary, not ground truth"
        record["maskAnnotation"] = "manual-mask-annotations.json"
        image = Image.open(args.output / record["image"]).convert("RGB")
        binary = working.point(lambda x: 255 if x >= 128 else 0)
        edge = np.asarray(binary.filter(ImageFilter.MaxFilter(7))) != np.asarray(binary.filter(ImageFilter.MinFilter(7)))
        overlay = np.asarray(image).copy()
        overlay[edge] = [255, 40, 80]
        overlay = Image.fromarray(overlay)
        cut = Image.composite(image, Image.new("RGB", image.size, "#222222"), binary)
        overlay.save(args.output / "comparisons" / (Path(record["name"]).stem + "-mask.jpg"))
        sheet.paste(overlay.resize((300, 400)), (i * 300, 32))
        sheet.paste(cut.resize((300, 400)), (i * 300, 432))
        ImageDraw.Draw(sheet).text((i * 300 + 8, 8), record["name"], fill="white")
        assert digest(Path(record["sourcePath"])) == record["sourceSHA256"]
    dataset["maskReview"] = annotations["limitations"]
    dataset["parentDatasetSHA256"] = digest(args.input / "dataset.json")
    (args.output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    sheet.save(args.output / "mask-review-v2.jpg")
    print(args.output / "mask-review-v2.jpg")


if __name__ == "__main__":
    main()
