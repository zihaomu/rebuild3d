#!/usr/bin/env python3
"""Assemble every original view, a clay turntable, and the preselected detail crops."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("regions", type=Path)
args = parser.parse_args()
root = args.run
records = json.loads((root / "inputs/dataset.json").read_text())["records"]
sheet = Image.new("RGB", (240 * len(records), 1050), "#222222")
draw = ImageDraw.Draw(sheet)
for index, record in enumerate(records):
    for row, path in enumerate([root / "inputs" / record["image"], root / f"renders/view-{index}-color.png",
                                root / f"renders/view-{index}-clay.png"]):
        image = Image.open(path).convert("RGB")
        image.thumbnail((234, 310))
        sheet.paste(image, (index * 240, row * 350 + 30))
        draw.text((index * 240 + 3, row * 350 + 6), ["photo", "texture", "clay"][row] + " " + record["name"], fill="white")
sheet.save(root / "review-all-views.jpg")
sheet = Image.new("RGB", (1280, 1005), "#222222")
draw = ImageDraw.Draw(sheet)
for index in range(12):
    image = Image.open(root / f"renders/turntable-clay-{index:02d}.png")
    image.thumbnail((320, 320))
    sheet.paste(image, ((index % 4) * 320, (index // 4) * 335 + 15))
    draw.text(((index % 4) * 320 + 3, (index // 4) * 335 + 2), str(index), fill="white")
sheet.save(root / "review-clay-turntable.jpg")
regions = json.loads(args.regions.read_text())["regions"]
sheet = Image.new("RGB", (800, len(regions) * 300), "#222222")
draw = ImageDraw.Draw(sheet)
for row, region in enumerate(regions):
    index = next(i for i, r in enumerate(records) if r["sourceSHA256"] == region["photoSHA256"])
    for column, path in enumerate([root / "inputs" / records[index]["image"], root / f"renders/view-{index}-color.png"]):
        image = Image.open(path).convert("RGB").crop(region["box"])
        image.thumbnail((390, 260))
        sheet.paste(image, (column * 400, row * 300 + 30))
        draw.text((column * 400 + 3, row * 300 + 5), region["id"] + (" photo" if column == 0 else " render"), fill="white")
sheet.save(root / "review-fixed-details.jpg")
print(f"Assembled all {len(records)} photo views and {len(regions)} fixed detail comparisons; visual acceptance is pending.")
