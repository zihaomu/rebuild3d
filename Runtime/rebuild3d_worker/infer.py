#!/usr/bin/env python3
"""Run verified local VGGT depth/camera inference for a managed sparse task."""
import argparse
from contextlib import contextmanager
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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import psutil
import torch
from PIL import Image
from .model_weights import (REPO_REVISION, WEIGHT_REVISION, WEIGHT_SHA256,
                            kernel_audit, load_model, verify_weights)
from .memory import footprint
from .failures import failure_fields


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
    report = {"status": "running", "stage": "verify", "samples": [], "timingsSeconds": {},
              "geometrySource": "learned inference; no measured geometry or verified multi-view support yet"}
    stop = threading.Event()
    backend = None
    backend_id = None

    @contextmanager
    def timed(name, gpu=False):
        report["stage"] = name
        if gpu and args.device == "mps":
            torch.mps.synchronize()
            if backend is not None:
                backend.synchronize()
        before = time.monotonic()
        try:
            yield
        finally:
            if gpu and args.device == "mps":
                torch.mps.synchronize()
                if backend is not None:
                    backend.synchronize()
            report["timingsSeconds"][name] = time.monotonic() - before

    def monitor():
        process = psutil.Process()
        with (args.output / "resources.jsonl").open("w") as stream:
            while not stop.is_set():
                memory = process.memory_info().rss
                driver = torch.mps.driver_allocated_memory() if args.device == "mps" else 0
                elapsed = time.monotonic() - start
                physical = footprint(os.getpid())
                row = {"seconds": elapsed, "rssBytes": memory, "mpsDriverBytes": driver,
                       "physicalFootprintBytes": physical, "stage": report["stage"]}
                if backend is not None:
                    row.update(backend.memory_stats())
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                usage = physical if physical is not None else max(memory, driver)
                if (args.max_seconds > 0 and elapsed > args.max_seconds) or usage > args.max_memory_gib * 1024**3:
                    report.update(status="resource-limit", elapsedSeconds=elapsed, resources=row)
                    write_json(args.output / "report.json", report)
                    os._exit(2)
                stop.wait(.25)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with timed("verify"):
            revision = verify_repository(args.repo)
            weight_info = verify_weights(args.weights)
            backend_id = weight_info['backendID']
        dataset = json.loads((args.dataset / "dataset.json").read_text())
        records = dataset["records"]
        assert 3 <= len(records) <= 12, "Current sparse backend supports 3–12 input photos"
        assert args.count is None or args.count == len(records), "Photo count mismatch; refusing silent truncation"
        args.count = len(records)
        config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "repoRevision": revision, "weightRevision": WEIGHT_REVISION,
                  "weightSHA256": weight_info["derivedWeightSHA256"], "weightFormat": weight_info,
                  "weightLicense": "CC-BY-NC-4.0; bundled for this local non-commercial build; see runtime NOTICE.md",
                  "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                  "seed": 0, "mpsFallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
                  "precision": weight_info.get("policy", "FP16 aggregator, FP32 heads on MPS; legacy CPU all FP32"),
                  "datasetSHA256": digest(args.dataset / "dataset.json"),
                  "photoIDs": [r["id"] for r in records], "photoNames": [r["name"] for r in records],
                  "sourceSHA256": [r["sourceSHA256"] for r in records]}
        config.update(backendID=weight_info["backendID"], modelImageSize=args.size)
        write_json(args.output / "config.json", config)
        torch.manual_seed(0)
        with timed("prepareInputs"):
            rgb, colors, masks, transforms = prepare(args.dataset, records, args.size, args.foreground)
        write_json(args.output / "pixel-transforms.json", transforms)
        for i, image in enumerate(rgb):
            Image.fromarray(image).save(args.output / f"input-{i}.jpg")
        report["stage"] = "load-weights"
        print("Loading verified checkpoint", flush=True)
        with timed("loadWeights", gpu=True):
            if weight_info["format"] == "rebuild3d-vggt-mlx":
                from .mlx_backend import MLXBackend
                backend = MLXBackend(args.repo, args.weights, weight_info, args.size, args.device)
                ignored = 0
                config.update(backend.config)
            else:
                model, ignored = load_model(args.repo, args.weights, args.device, info=weight_info)
        config["intentionallyUnusedPointAndTrackWeights"] = ignored
        write_json(args.output / "config.json", config)
        dtype = torch.float16 if args.device == "mps" else torch.float32
        images = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to(args.device, dtype=dtype)[None] / 255
        report["stage"] = "aggregate"
        print(f"Aggregate {args.count} images at {args.size} pixels", flush=True)
        if backend is not None:
            pose, depth, confidence, audit = backend.predict(
                images, timed, lambda stage: print("Aggregate " + stage, flush=True))
            report["packedKernelCalls"] = audit["packedKernelCalls"]
            report["executedLowBitParameterCount"] = audit["executedLowBitParameterCount"]
            if report["executedLowBitParameterCount"] != weight_info["lowBitParameterCount"]:
                raise ValueError("Not all declared low-bit parameters executed")
        else:
            with torch.inference_mode():
                with timed("aggregator", gpu=True):
                    tokens, patch_start = model.aggregator(images)
                with timed("headInputConversion", gpu=True):
                    tokens = [t.float() if t is not None else None for t in tokens]
                report["stage"] = "camera-depth"
                print("Predict cameras and depth", flush=True)
                with timed("cameraHeadAndReadback", gpu=True):
                    pose = model.camera_head(tokens)[-1].float().cpu()
                with timed("depthHeadAndReadback", gpu=True):
                    depth, confidence = model.depth_head(tokens, images.float(), patch_start, frames_chunk_size=1)
                    depth = depth.float().cpu().numpy()[0, ..., 0]
                    confidence = confidence.float().cpu().numpy()[0]
            audit = kernel_audit(model)
            report["packedKernelCalls"] = sum(m["calls"] for m in audit.values())
            report["executedLowBitParameterCount"] = sum(m["originalWeightParameters"] for m in audit.values() if m["calls"])
        write_json(args.output / "kernel-audit.json", audit)
        output_start = time.monotonic()
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
        report["timingsSeconds"]["serializeOutputs"] = time.monotonic() - output_start
        report.update(status="inference-complete", stage="saved", elapsedSeconds=time.monotonic() - start,
                      cameraCount=len(records), depthRange=[float(depth.min()), float(depth.max())],
                      next="Export and inspect geometry; inference completion is not reconstruction acceptance")
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(**failure_fields(error, backend_id), traceback=traceback.format_exc(),
                      elapsedSeconds=time.monotonic() - start)
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
