#!/usr/bin/env python3
"""Package a reviewed approximation with source identities and hashed provenance artifacts."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("fusion", type=Path)
    p.add_argument("usdz", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    dataset = json.loads((a.dataset / "dataset.json").read_text())
    report = json.loads((a.fusion / "report.json").read_text())
    sources = [{"name": r["name"], "sha256": r["sourceSHA256"]} for r in dataset["records"]]
    for r in dataset["records"]:
        assert sha(Path(r["sourcePath"])) == r["sourceSHA256"]
    files = {"model.usdz": a.usdz / "model.usdz", "provenance.usdz": a.usdz / "provenance.usdz",
             "model.glb": a.fusion / "model.glb", "provenance.glb": a.fusion / "provenance.glb",
             "without-completion.glb": a.fusion / "without-completion.glb",
             "face-provenance.npz": a.fusion / "face-provenance.npz", "fusion.json": a.fusion / "report.json",
             "review-metrics.json": a.fusion / "review-metrics.json", "seven-view-review.jpg": a.fusion / "seven-view-review.jpg",
             "untextured-turntable.gif": a.fusion / "untextured-turntable.gif",
             "inference-config.json": a.run / "config.json", "cameras.json": a.run / "cameras.json",
             "pixel-transforms.json": a.run / "pixel-transforms.json", "dataset.json": a.dataset / "dataset.json",
             "mask-annotations.json": a.dataset / "manual-mask-annotations.json"}
    artifacts = []
    for name, source in files.items():
        destination = a.output / name
        shutil.copy2(source, destination)
        artifacts.append({"path": name, "sha256": sha(destination), "byteCount": destination.stat().st_size})
    bundle = {"formatVersion": 1, "kind": "approximate", "method": "VGGT-1B learned camera/depth + silhouette-constrained TSDF",
              "createdAt": datetime.now(timezone.utc).isoformat(), "sourcePhotos": sources,
              "model": "model.usdz", "provenanceModel": "provenance.usdz", "artifacts": artifacts,
              "triangleCount": report["triangles"], "completionTriangleCount": report["completionTriangles"],
              "limitations": ["All geometry and cameras are inferred, not measured; relative scale only.",
                              "Orange regions: learned depth fusion. Purple regions: silhouette-based completion.",
                              "Face details, thin instrument parts and occluded regions remain approximate.",
                              "Seven original-view checks are input consistency, not ground-truth 3D accuracy.",
                              "Research uses original VGGT-1B CC-BY-NC-4.0 weights; no weights bundled in the application."]}
    (a.output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(a.output / "bundle.json")


if __name__ == "__main__":
    main()
