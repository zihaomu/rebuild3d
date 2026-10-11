#!/usr/bin/env python3
"""Freeze inputs/components and measure fresh, serial reconstruction jobs on macOS.

Run with the managed runtime's Python. This does not modify the baseline runtime,
reuse task checkpoints, or claim that a backend measurement includes application UI.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def command_output(*command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    return {"command": list(command), "exitCode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def environment():
    return {"atUTC": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": sys.version, "cpuCount": psutil.cpu_count(),
            "memoryBytes": psutil.virtual_memory().total, "swap": psutil.swap_memory()._asdict(),
            "loadAverage": os.getloadavg(), "hardware": command_output("sysctl", "-n", "hw.model"),
            "power": command_output("pmset", "-g", "batt"),
            "thermal": command_output("pmset", "-g", "therm"),
            "processes": command_output("ps", "-axo", "pid,ppid,%cpu,rss,comm")}


# sys/resource.h rusage_info_v2; libproc accounts physical footprint separately
# from resident pages and GPU driver allocation. Never add these metrics together.
class RUsageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
            "proc_exit_abstime", "child_user_time", "child_system_time", "child_pkg_idle_wkups",
            "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
            "diskio_bytesread", "diskio_byteswritten")]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None
if _libproc:
    _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _libproc.proc_pid_rusage.restype = ctypes.c_int


def footprint(pid):
    if not _libproc:
        return None
    info = RUsageV2()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:
        return None
    return int(info.phys_footprint)


def sample_tree(pid):
    rows = []
    try:
        processes = [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        return rows
    for process in processes:
        try:
            rows.append({"pid": process.pid, "rssBytes": process.memory_info().rss,
                         "physicalFootprintBytes": footprint(process.pid),
                         "cpuSeconds": sum(process.cpu_times()[:2]), "name": process.name()})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            rows.append({"pid": process.pid, "rssBytes": None,
                         "physicalFootprintBytes": None, "samplingError": repr(error)})
    return rows


def memory_pressure():
    """macOS pressure level (1 normal, 2 warning, 4 critical), or unavailable."""
    if sys.platform != "darwin":
        return None
    value = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    libc = ctypes.CDLL(None)
    if libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value),
                         ctypes.byref(size), None, 0) != 0:
        return None
    return value.value


def freeze(args):
    output, runtime = args.output.resolve(), args.runtime.resolve()
    output.mkdir(parents=True, exist_ok=False)
    registry = json.loads((ROOT / "doc/v3-测试输入清单.json").read_text())
    runtime_manifest = json.loads((runtime / "runtime.json").read_text())
    for index, item in enumerate(runtime_manifest["artifacts"]):
        path = runtime / item["path"]
        if not path.resolve().is_relative_to(runtime) or not path.is_file():
            raise ValueError("Missing/outside runtime file: " + str(path))
        if path.stat().st_size != item["byteCount"] or sha(path) != item["sha256"]:
            raise ValueError("Changed baseline component: " + str(path))
        if index % 5000 == 0:
            print("Verified runtime files", index, flush=True)
    preserved = json.loads((ROOT / "build/v3-oneclick/baseline/preserved-artifacts.json").read_text())
    for item in preserved["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("Historical artifact changed: " + str(path))
    save(output / "preserved-artifacts.json", preserved)
    tasks = {}
    for dataset in registry["sets"]:
        label = dataset["id"]
        photos = json.loads((ROOT / f"build/v3-oneclick/runs/{label}-managed/photos.json").read_text())
        by_name = {p["name"]: p for p in photos}
        if len(photos) != 7 or len(by_name) != 7:
            raise ValueError("Seven unique photo identities required")
        for photo in dataset["files"]:
            source = (ROOT / dataset["source_directory"] / photo["filename"]).resolve()
            if sha(source) != photo["sha256"] or source.stat().st_size != photo["byte_count"]:
                raise ValueError("Original changed: " + str(source))
            record = by_name[photo["filename"]]
            if record["sourceSHA256"] != photo["sha256"]:
                raise ValueError("Old photo identity disagrees with registry")
            record["sourcePath"] = str(source)
        tasks[label] = {
            "formatVersion": 1, "jobID": str(uuid.uuid4()), "photos": photos,
            "runtime": {"nativeWorker": str(runtime / "rebuild3d-native-worker"),
                        "rasterLibrary": str(runtime / "worker/rebuild3d_worker/raster.dylib"),
                        "vggtRepository": str(runtime / "vggt"),
                        "weights": str(runtime / "models/vggt-1b.safetensors")},
            "parameters": {"memoryGiB": 10.5, "modelSize": 518,
                           "volumeResolution": 256, "textureSize": 4096}}
        save(output / f"{label}-task-template.json", tasks[label])
        regions = ROOT / f"build/v3-oneclick/baseline/{label}-review-regions.json"
        (output / f"{label}-review-regions.json").write_bytes(regions.read_bytes())
    save(output / "protocol.json", {
        "formatVersion": 1, "frozenAtUTC": datetime.now(timezone.utc).isoformat(),
        "baselineCommit": command_output("git", "rev-parse", "HEAD")["stdout"],
        "baselineRuntime": str(runtime), "runtimeManifestSHA256": sha(runtime / "runtime.json"),
        "verifiedRuntimeFileCount": len(runtime_manifest["artifacts"]),
        "originalPhotoCount": 14, "verifiedHistoricalFiles": len(preserved["files"]),
        "thresholds": {"rotationP95Degrees": 1, "focalRelativeP95": .02,
                       "depthRelativeMedian": .02, "depthRelativeP95": .05,
                       "validDepthFractionDrop": .005, "silhouetteIoUDrop": .01,
                       "photoTexelCoverageDrop": .02},
        "comparison": "One scene-wide similarity only; baseline is inferred geometry, not ground truth.",
        "datasets": {"A": "required regression, no calibration", "B": "required regression, no calibration",
                     "C": "existing conventional regression", "D": "existing skull regression",
                     "Q": "not needed for first data-free weight-only experiment; freeze separately before calibration",
                     "E": "source/identity must be frozen before final configuration; never use for tuning"},
        "resourceSamplingSeconds": .25,
        "physicalMemoryMetric": "libproc rusage_info_v2 physical footprint, per-process and process-tree ledger sum; not added to RSS or GPU allocation",
        "finalPerformance": "At least three interleaved fresh tasks per baseline/candidate and A/B; UI timing separate from backend.",
        "environment": environment()})
    print("Frozen", output, flush=True)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    manifest["jobID"] = str(uuid.uuid4())
    task = output / "job"
    task.mkdir()
    save(output / "manifest.json", manifest)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(args.worker_root.resolve()), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", REBUILD3D_PARENT_PID=str(os.getpid()))
    # Preserve a venv's executable path: resolving its symlink selects the base
    # Python and silently loses the experimental runtime's installed packages.
    command = [str(args.python.absolute()), "-m", "rebuild3d_worker.worker",
               str(output / "manifest.json"), str(task)]
    before = environment()
    started = time.monotonic()
    with (output / "worker.log").open("w") as log, (output / "process-resources.jsonl").open("w") as resources:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        save(output / "process.json", {"pid": process.pid, "command": command,
                                       "startedAtUTC": before["atUTC"], "label": args.label})
        peak_rss = peak_footprint = 0
        incomplete_footprint_samples = 0
        def stop(signum, _):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while process.poll() is None:
            rows = sample_tree(process.pid)
            rss = (sum(r["rssBytes"] for r in rows)
                   if rows and all(r["rssBytes"] is not None for r in rows) else None)
            physical = (sum(r["physicalFootprintBytes"] for r in rows)
                        if rows and all(r["physicalFootprintBytes"] is not None for r in rows) else None)
            incomplete_footprint_samples += physical is None
            if rss is not None:
                peak_rss = max(peak_rss, rss)
            if physical is not None:
                peak_footprint = max(peak_footprint, physical)
            sample = {"atUTC": datetime.now(timezone.utc).isoformat(),
                      "seconds": time.monotonic() - started, "processes": rows,
                      "treeRSSBytes": rss, "treePhysicalFootprintBytes": physical,
                      "swapUsedBytes": psutil.swap_memory().used, "systemCPUPercent": psutil.cpu_percent(),
                      "memoryPressureLevel": memory_pressure(),
                      "availableMemoryBytes": psutil.virtual_memory().available}
            resources.write(json.dumps(sample) + "\n")
            resources.flush()
            time.sleep(.25)
        exit_code = process.wait()
    elapsed = time.monotonic() - started
    report = {"label": args.label, "exitCode": exit_code, "elapsedSeconds": elapsed,
              "kind": "fresh backend task; excludes UI and initial component installation",
              "sampledPeakTreeRSSBytes": peak_rss or None,
              "sampledPeakTreePhysicalFootprintBytes": peak_footprint or None,
              "incompleteFootprintSamples": incomplete_footprint_samples,
              "before": before, "after": environment(), "stageEvents": []}
    events = task / "events.jsonl"
    if events.exists():
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        report["stageEvents"] = [r for r in rows if r["event"] in ("stage-completed", "completed", "failed")]
        report["reusedStageCount"] = sum(r["event"] == "stage-reused" for r in rows)
    save(output / "benchmark.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("before", "after", "stageEvents")}), flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    f = commands.add_parser("freeze")
    f.add_argument("output", type=Path)
    f.add_argument("--runtime", type=Path, required=True)
    r = commands.add_parser("run")
    r.add_argument("manifest", type=Path)
    r.add_argument("output", type=Path)
    r.add_argument("--python", type=Path, required=True)
    r.add_argument("--worker-root", type=Path, required=True)
    r.add_argument("--label", required=True)
    args = parser.parse_args()
    return freeze(args) if args.action == "freeze" else run(args)


if __name__ == "__main__":
    sys.exit(main())
