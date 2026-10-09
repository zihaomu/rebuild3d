"""Create a portable photo-textured result and separate geometry/appearance source views."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .common import sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run, output = args.run, args.output
    if output.exists():
        raise ValueError("Result output must be new")
    shutil.copytree(run / "texture", output)
    dataset = json.loads((run / "inputs/dataset.json").read_text())
    inputs = json.loads((run / "input-use.json").read_text())
    for photo in inputs["allPhotos"]:
        assert sha(photo["sourcePath"]) == photo["sourceSHA256"], "An original photo changed"
        photo.pop("sourcePath")
    write_json(output / "input-use.json", inputs)
    for record in dataset["records"]:
        assert sha(record["sourcePath"]) == record["sourceSHA256"], "Source photo changed"
        for key in ("image", "mask", "fullMask"):
            source = run / "inputs" / record[key]
            destination = output / record[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        # Resolve originals by identity in the enclosing project after relocation.
        record.pop("sourcePath")
        record["originalLocator"] = "project photo ID and sourceSHA256"
    write_json(output / "dataset.json", dataset)
    provenance = json.loads((output / "texture-provenance.json").read_text())
    provenance["sourcePhotos"] = dataset["records"]
    write_json(output / "texture-provenance.json", provenance)
    for source, destination in [
        ("inference/pixel-transforms.json", "pixel-transforms.json"),
        ("inference/config.json", "inference-config.json"),
        ("fusion/face-provenance.npz", "geometry-provenance.npz"),
        ("uv/report.json", "uv-report.json"),
    ]:
        shutil.copy2(run / source, output / destination)
    geometry = np.load(output / "geometry.npz")
    face_ids = np.load(output / "atlas-surface.npz")["originalFaceID"]
    active = face_ids >= 0
    colors = np.zeros((*face_ids.shape, 3), dtype="uint8")
    classes = geometry["completion"][face_ids[active]]
    colors[active] = np.where(classes[:, None], [197, 88, 235], [244, 160, 42])
    distance, nearest = distance_transform_edt(~active, return_indices=True)
    padding = (~active) & (distance <= 4)
    colors[padding] = colors[nearest[0][padding], nearest[1][padding]]
    Image.fromarray(colors).save(output / "geometry-source.png")
    texture_sources = np.asarray(Image.open(output / "texture-source-view.png")).copy()
    texture_sources[padding] = texture_sources[nearest[0][padding], nearest[1][padding]]
    Image.fromarray(texture_sources).save(output / "texture-source-display.png")
    for name, texture in [("model", "basecolor.png"), ("provenance", "geometry-source.png"),
                          ("texture-sources", "texture-source-display.png")]:
        subprocess.run([sys.executable, "-m", "rebuild3d_worker.export", str(output),
                        "--name", name, "--texture", texture], check=True)
    fusion = json.loads((output / "fusion.json").read_text())
    report = json.loads((output / "report.json").read_text())
    limitations = [
        "All geometry and cameras are inferred, not measured. Scale is relative.",
        "Orange indicates learned depth; purple indicates silhouette completion.",
        "Texture source colors identify photographs; grey is low-frequency appearance fill.",
        "Photo-supported texels do not establish calibrated 3D accuracy or recovered albedo.",
        "Unseen surfaces, thin parts, residual seams and shadows can remain approximate.",
    ]
    (output / "README.md").write_text(
        "# 自动近似重建来源包\n\n"
        "`model.usdz` / `model.glb`：内嵌原照片颜色。\n\n"
        "`provenance.usdz`：橙色为学习深度推测，紫色为轮廓补全，全部几何仍为推测。\n\n"
        "`texture-sources.usdz`：照片来源色见 `texture-source-view.png` 和 "
        "`texture-provenance.json`；灰色为外观填充，不是观测到的新细节。\n\n"
        f"有效 UV 像素的照片取色比例：{report['photoCoverageTexelFraction']:.2%}。"
        "源照片通过照片 ID 和 SHA-256 在项目中查找；请保留整个项目或导出来源包。\n\n"
        "此包的结构校验不等于视觉验收；有限照片的形状和遮挡处可能粗糙。\n"
    )
    artifacts = [{"path": str(p.relative_to(output)), "sha256": sha(p), "byteCount": p.stat().st_size}
                 for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "bundle.json", {
        "formatVersion": 1, "kind": "approximate", "method": "Automatic VGGT geometry and photographic UV texture",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourcePhotos": [{"name": r["name"], "sha256": r["sourceSHA256"]} for r in inputs["allPhotos"]],
        "model": "model.usdz", "provenanceModel": "provenance.usdz", "artifacts": artifacts,
        "triangleCount": fusion["triangles"], "completionTriangleCount": fusion["completionTriangles"],
        "limitations": limitations,
    })
    print(json.dumps({"status": "structurally-validated-candidate", "triangles": fusion["triangles"],
                      "photoCoverage": report["photoCoverageTexelFraction"], "visualAcceptance": "pending"}), flush=True)


if __name__ == "__main__":
    main()
