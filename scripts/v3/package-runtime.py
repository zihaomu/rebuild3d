#!/usr/bin/env python3
"""Build local runtime resources from verified development components; never runs on the user's generation path."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vggt", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("Runtime output must be new; installed runtimes are immutable")
    revision = subprocess.check_output(["git", "-C", str(args.vggt), "rev-parse", "HEAD"], text=True).strip()
    if revision != "a288dd0f14786c93483e45524328726ab7b1b4ce":
        raise ValueError("Unexpected VGGT revision")
    weight_sha = digest(args.weights)
    if weight_sha != "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e":
        raise ValueError("Unexpected VGGT weights")
    info = json.loads(subprocess.check_output([str(args.python), "-c",
        "import sys,sysconfig,json;print(json.dumps(dict(prefix=sys.base_prefix,packages=sysconfig.get_path('purelib'),version=sys.version)))"], text=True))
    source_prefix, source_packages = Path(info["prefix"]), Path(info["packages"])
    if not (source_prefix / "bin/python3.12").is_file():
        raise ValueError("Expected the verified standalone CPython 3.12 distribution")
    destination.mkdir(parents=True)
    # Dereference development symlinks; preserve no dependency on uv, Homebrew or the venv.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(source_prefix, destination / "python", symlinks=False, ignore=ignore)
    site = destination / "python/lib/python3.12/site-packages"
    shutil.copytree(source_packages, site, dirs_exist_ok=True, symlinks=False, ignore=ignore)
    shutil.copytree(root / "Runtime/rebuild3d_worker", destination / "worker/rebuild3d_worker", ignore=ignore)
    shutil.copytree(args.vggt / "vggt", destination / "vggt/vggt", ignore=ignore)
    shutil.copy2(args.vggt / "LICENSE.txt", destination / "vggt/LICENSE.txt")
    source_files = {str(p.relative_to(destination / "vggt")): digest(p)
                    for p in sorted((destination / "vggt").rglob("*")) if p.is_file()}
    (destination / "vggt/source-manifest.json").write_text(json.dumps({"revision": revision, "files": source_files}, indent=2) + "\n")
    (destination / "models").mkdir()
    shutil.copy2(args.weights, destination / "models/vggt-1b.safetensors")
    shutil.copy2(args.native, destination / "rebuild3d-native-worker")
    subprocess.run(["clang++", "-O3", "-std=c++17", "-dynamiclib", str(root / "Runtime/rebuild3d_worker/raster.cpp"),
                    "-o", str(destination / "worker/rebuild3d_worker/raster.dylib")], check=True)
    # Dependency dist-info and embedded notices are retained with their complete wheels.
    shutil.copy2(root / "scripts/texture/requirements-macos.lock", destination / "requirements-macos.lock")
    (destination / "NOTICE.md").write_text(
        "# Local inference components\n\n"
        "VGGT code is pinned; see vggt/LICENSE.txt and vggt/source-manifest.json. "
        "VGGT-1B weights revision 860abec7937da0a4c03c41d3c269c366e82abdf9 is labelled CC-BY-NC-4.0 "
        "by its original model card: https://huggingface.co/facebook/VGGT-1B/blob/860abec7937da0a4c03c41d3c269c366e82abdf9/README.md . "
        "This local build does not establish permission for commercial redistribution.\n\n"
        "CPython and wheel dependency notices are retained inside python/. No input photographs are included.\n"
    )
    files = [{"path": str(p.relative_to(destination)), "byteCount": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(destination.rglob("*")) if p.is_file()]
    manifest = {"formatVersion": 1, "platform": "macos-arm64", "pythonVersion": info["version"],
                "vggtRevision": revision, "weightSHA256": weight_sha, "artifacts": files}
    (destination / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "artifacts": len(files), "bytes": sum(p["byteCount"] for p in files)}))


if __name__ == "__main__":
    main()
