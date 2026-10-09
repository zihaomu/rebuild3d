#!/usr/bin/env python3
"""Run a pinned, local VGGT depth/camera experiment; no photos leave the machine."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import psutil
import torch
from PIL import Image
from safetensors import safe_open

REPO_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
WEIGHT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
WEIGHT_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_repository(repo):
    manifest = repo / "source-manifest.json"
    if manifest.is_file():
        source = json.loads(manifest.read_text())
        if source["revision"] != REPO_REVISION:
            raise ValueError("Unexpected packaged VGGT revision")
        actual = {str(p.relative_to(repo)) for p in (repo / "vggt").rglob("*.py")}
        expected = {p for p in source["files"] if p.endswith(".py")}
        if actual != expected:
            raise ValueError("Packaged VGGT source list changed")
        for relative, checksum in source["files"].items():
            path = (repo / relative).resolve()
            if not path.is_relative_to(repo.resolve()) or digest(path) != checksum:
                raise ValueError("Packaged VGGT source changed: " + relative)
        return source["revision"]
    # Development checkout only. Shipped components always carry a verified manifest.
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != REPO_REVISION:
        raise ValueError("Unexpected development VGGT revision")
    return revision


def prepare(dataset_root, records, size, foreground):
    images, colors, masks, transforms = [], [], [], []
    for record in records:
        source = dataset_root / record["image"]
        mask_path = dataset_root / record["mask"]
        assert digest(source) == record["workingSHA256"]
        assert digest(mask_path) == record["maskSHA256"]
        assert digest(Path(record["sourcePath"])) == record["sourceSHA256"]
        image = Image.open(source).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        w, h = image.size
        scale = size / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (size - nw) // 2, (size - nh) // 2
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        mask = mask.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(image, (left, top))
        mask_canvas = Image.new("L", (size, size))
        mask_canvas.paste(mask, (left, top))
        rgb = np.asarray(canvas).copy()
        binary = np.asarray(mask_canvas) >= 128
        colors.append(rgb)
        masks.append(binary)
        if foreground:
            rgb[~binary] = 255
        images.append(rgb)
        sx, sy = nw / w, nh / h
        transform = np.array([[sx, 0, left + (sx - 1) / 2],
                              [0, sy, top + (sy - 1) / 2], [0, 0, 1]])
        encoded = np.asarray(record["encodedToWorkingPixels"]).reshape(3, 3)
        transforms.append({"name": record["name"], "photoID": record["id"], "workingToModelPixels": transform.tolist(),
                           "encodedToModelPixels": (transform @ encoded).tolist(),
                           "resizedSize": [nw, nh], "paddingLeftTop": [left, top]})
    return np.stack(images), np.stack(colors), np.stack(masks), transforms


def load_model(repo, weights, device):
    sys.path.insert(0, str(repo.resolve()))
    from vggt.models.vggt import VGGT
    # Only the constructor's stochastic-depth scalar schedule needs real CPU data.
    # Parameters are allocated on meta, then loaded one tensor at a time to avoid
    # simultaneously holding full FP32 and FP16 copies on a 16 GB Mac.
    linspace = torch.linspace
    with patch("torch.linspace", lambda *a, **k: linspace(*a, **{**k, "device": "cpu"})):
        with torch.device("meta"):
            model = VGGT(enable_point=False, enable_track=False)
    parameters = dict(model.named_parameters())
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        missing = set(parameters) - available
        extra = available - set(parameters)
        assert not missing, f"Missing weights: {sorted(missing)}"
        assert all(k.startswith(("point_head.", "track_head.")) for k in extra), sorted(extra)
        for name, parameter in parameters.items():
            tensor = checkpoint.get_tensor(name)
            assert tensor.shape == parameter.shape, name
            dtype = torch.float16 if name.startswith("aggregator.") and device == "mps" else torch.float32
            tensor = tensor.to(device=device, dtype=dtype)
            owner, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(owner), leaf, torch.nn.Parameter(tensor, requires_grad=False))
    dtype = torch.float16 if device == "mps" else torch.float32
    for name, values in (("_resnet_mean", [.485, .456, .406]), ("_resnet_std", [.229, .224, .225])):
        setattr(model.aggregator, name, torch.tensor(values, device=device, dtype=dtype).view(1, 1, 3, 1, 1))
    assert not any(p.is_meta for p in model.parameters())
    return model.eval(), len(extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--count", type=int, help="Optional exact count assertion; never truncates input")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--max-seconds", type=int, default=0, help="0 disables elapsed-time limit")
    parser.add_argument("--max-memory-gib", type=float, default=11)
    args = parser.parse_args()
    assert args.size % 14 == 0
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"status": "running", "stage": "verify", "samples": [],
              "geometrySource": "learned inference; no measured geometry or verified multi-view support yet"}
    stop = threading.Event()

    def monitor():
        process = psutil.Process()
        with (args.output / "resources.jsonl").open("w") as stream:
            while not stop.is_set():
                memory = process.memory_info().rss
                driver = torch.mps.driver_allocated_memory() if args.device == "mps" else 0
                elapsed = time.monotonic() - start
                stream.write(json.dumps({"seconds": elapsed, "rssBytes": memory, "mpsDriverBytes": driver}) + "\n")
                stream.flush()
                if (args.max_seconds > 0 and elapsed > args.max_seconds) or max(memory, driver) > args.max_memory_gib * 1024**3:
                    report.update(status="resource-limit", elapsedSeconds=elapsed, rssBytes=memory, mpsDriverBytes=driver)
                    write_json(args.output / "report.json", report)
                    os._exit(2)
                stop.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        revision = verify_repository(args.repo)
        assert digest(args.weights) == WEIGHT_SHA256, "Checkpoint SHA-256 mismatch"
        dataset = json.loads((args.dataset / "dataset.json").read_text())
        records = dataset["records"]
        assert 3 <= len(records) <= 12, "Current sparse backend supports 3–12 input photos"
        assert args.count is None or args.count == len(records), "Photo count mismatch; refusing silent truncation"
        args.count = len(records)
        config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "repoRevision": revision, "weightRevision": WEIGHT_REVISION, "weightSHA256": WEIGHT_SHA256,
                  "weightLicense": "CC-BY-NC-4.0; bundled for this local non-commercial build; see runtime NOTICE.md",
                  "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                  "seed": 0, "mpsFallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
                  "precision": "FP16 aggregator, FP32 camera and depth heads on MPS; CPU all FP32",
                  "datasetSHA256": digest(args.dataset / "dataset.json"),
                  "photoIDs": [r["id"] for r in records], "photoNames": [r["name"] for r in records],
                  "sourceSHA256": [r["sourceSHA256"] for r in records]}
        write_json(args.output / "config.json", config)
        torch.manual_seed(0)
        rgb, colors, masks, transforms = prepare(args.dataset, records, args.size, args.foreground)
        write_json(args.output / "pixel-transforms.json", transforms)
        for i, image in enumerate(rgb):
            Image.fromarray(image).save(args.output / f"input-{i}.jpg")
        report["stage"] = "load-weights"
        print("Loading verified checkpoint", flush=True)
        model, ignored = load_model(args.repo, args.weights, args.device)
        config["intentionallyUnusedPointAndTrackWeights"] = ignored
        write_json(args.output / "config.json", config)
        dtype = torch.float16 if args.device == "mps" else torch.float32
        images = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to(args.device, dtype=dtype)[None] / 255
        report["stage"] = "aggregate"
        print(f"Aggregate {args.count} images at {args.size} pixels", flush=True)
        with torch.inference_mode():
            tokens, patch_start = model.aggregator(images)
            tokens = [t.float() if t is not None else None for t in tokens]
            report["stage"] = "camera-depth"
            print("Predict cameras and depth", flush=True)
            pose = model.camera_head(tokens)[-1].float().cpu()
            depth, confidence = model.depth_head(tokens, images.float(), patch_start, frames_chunk_size=1)
            depth = depth.float().cpu().numpy()[0, ..., 0]
            confidence = confidence.float().cpu().numpy()[0]
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, (args.size, args.size))
        extrinsics, intrinsics = extrinsics.numpy()[0], intrinsics.numpy()[0]
        assert all(np.isfinite(a).all() for a in (depth, confidence, extrinsics, intrinsics))
        np.savez_compressed(args.output / "predictions.npz", depth=depth, confidence=confidence,
                            extrinsics=extrinsics, intrinsics=intrinsics, rgb=colors, masks=masks)
        cameras = [{"name": r["name"], "photoID": r["id"], "source": "VGGT learned estimate; not yet independently verified",
                    "worldToCamera": e.tolist(), "modelIntrinsics": k.tolist(),
                    "workingIntrinsics": (np.linalg.inv(t["workingToModelPixels"]) @ k).tolist()}
                   for r, e, k, t in zip(records, extrinsics, intrinsics, transforms)]
        write_json(args.output / "cameras.json", {"convention": "OpenCV: right, down, forward; relative scale", "cameras": cameras})
        report.update(status="inference-complete", stage="saved", elapsedSeconds=time.monotonic() - start,
                      cameraCount=len(records), depthRange=[float(depth.min()), float(depth.max())],
                      next="Export and inspect geometry; inference completion is not reconstruction acceptance")
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc(), elapsedSeconds=time.monotonic() - start)
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
