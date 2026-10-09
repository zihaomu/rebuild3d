"""Versioned, resumable local task supervisor. Each heavy stage runs in a fresh process."""
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import uuid

import psutil


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_manifest(directory, root):
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "resources.jsonl"]


def checkpoint_valid(checkpoint, signature, root):
    if not isinstance(checkpoint, dict) or checkpoint.get("signature") != signature or not checkpoint.get("artifacts"):
        return False
    if not isinstance(checkpoint["artifacts"], list):
        return False
    for item in checkpoint["artifacts"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not {"bytes", "sha256"} <= item.keys():
            return False
        path = root / item["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            return False
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            return False
    return True


class Cancelled(Exception):
    pass


class Supervisor:
    def __init__(self, manifest, root):
        self.manifest, self.root = manifest, root
        self.cancelled = False
        self.owner_pid = int(os.environ.get("REBUILD3D_PARENT_PID", "0"))
        self.started = time.monotonic()
        self.state = {"protocolVersion": 1, "jobID": manifest["jobID"], "status": "starting"}
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("checkpoints", "logs", "attempts"):
            (root / directory).mkdir(exist_ok=True)

    def emit(self, event, stage=None, **values):
        record = {"protocolVersion": 1, "jobID": self.manifest["jobID"], "event": event,
                  "stage": stage, "elapsedSeconds": time.monotonic() - self.started,
                  "timestamp": datetime.now(timezone.utc).isoformat(), **values}
        line = json.dumps(record, ensure_ascii=False)
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")
        try:
            print(line, flush=True)
        except BrokenPipeError:
            self.cancelled = True

    def save_state(self, **values):
        self.state.update(values)
        atomic_json(self.root / "state.json", self.state)

    def stop(self, *_):
        self.cancelled = True

    def check_cancelled(self):
        # The app owns this supervisor. A force-quit must stop native/GPU work too.
        if self.owner_pid and os.getppid() != self.owner_pid:
            self.cancelled = True
        if self.cancelled:
            raise Cancelled()

    def execute(self, stage, command, environment, memory_limit):
        self.check_cancelled()
        self.save_state(status="running", stage=stage)
        self.emit("stage-started", stage)
        started, peak = time.monotonic(), 0
        with (self.root / "logs" / (stage + ".log")).open("a") as log:
            guarded = [sys.executable, "-m", "rebuild3d_worker.stage_guard", str(os.getpid()), *command]
            process = subprocess.Popen(guarded, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            pending = b""
            try:
                with (self.root / "logs" / (stage + "-resources.jsonl")).open("a") as resources:
                    while True:
                        self.check_cancelled()
                        try:
                            parent = psutil.Process(process.pid)
                            rss = 0
                            for item in [parent] + parent.children(recursive=True):
                                try:
                                    rss += item.memory_info().rss
                                except psutil.NoSuchProcess:
                                    pass
                            peak = max(peak, rss)
                            resources.write(json.dumps({"seconds": time.monotonic() - started, "rssBytes": rss}) + "\n")
                            resources.flush()
                            if rss > memory_limit:
                                raise RuntimeError(f"Stage {stage} exceeded the task memory budget")
                        except psutil.NoSuchProcess:
                            pass
                        for key, _ in selector.select(timeout=1):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            pending += chunk
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                text = line.decode("utf-8", errors="replace")
                                log.write(text + "\n"); log.flush()
                                try:
                                    data = json.loads(text)
                                except json.JSONDecodeError:
                                    data = None
                                if isinstance(data, dict) and data.get("event") == "progress":
                                    self.emit("progress", stage, completed=data.get("completed"),
                                              total=data.get("total"), photo=data.get("photo"))
                                else:
                                    self.emit("diagnostic", stage, message=text[-2000:])
                        if process.poll() is not None and not selector.get_map():
                            break
                if pending:
                    log.write(pending.decode("utf-8", errors="replace")); log.flush()
                if process.returncode != 0:
                    raise RuntimeError(f"Stage {stage} failed (exit {process.returncode}); see logs/{stage}.log")
            finally:
                selector.close()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                process.stdout.close()
        return {"stageElapsedSeconds": time.monotonic() - started, "sampledProcessTreePeakRSSBytes": peak}

    def run(self):
        manifest, root = self.manifest, self.root
        if manifest.get("formatVersion") != 1:
            raise ValueError("Unsupported task protocol")
        photos = manifest["photos"]
        if not 3 <= len(photos) <= 12:
            raise ValueError("Sparse task currently supports 3–12 photos; none will be silently dropped")
        if len({p["id"] for p in photos}) != len(photos) or len({p["sourceSHA256"] for p in photos}) != len(photos):
            raise ValueError("Photo IDs and source contents must be unique")
        all_photos = manifest.get("allPhotos", photos)
        if len({p["id"] for p in all_photos}) != len(all_photos) or len({p["sourceSHA256"] for p in all_photos}) != len(all_photos):
            raise ValueError("All input photo identities must be unique")
        if not all(p in all_photos for p in photos):
            raise ValueError("Selected photos must exactly match their original identities")
        if len(all_photos) > len(photos):
            uses = manifest.get("photoUse", [])
            if {p["photoID"] for p in uses} != {p["id"] for p in all_photos}:
                raise ValueError("Every input needs a recorded purpose when selecting a subset")
        for photo in all_photos:
            if sha(photo["sourcePath"]) != photo["sourceSHA256"]:
                raise ValueError("Input changed: " + photo["name"])
        runtime = manifest["runtime"]
        parameters = manifest.get("parameters", {})
        native = Path(runtime["nativeWorker"]).resolve()
        raster = Path(runtime["rasterLibrary"]).resolve()
        code = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
        repository = Path(runtime["vggtRepository"]).resolve()
        model_code = {str(p.relative_to(repository)): sha(p) for p in (repository / "vggt").rglob("*.py")}
        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "numpy", "pillow", "scipy", "scikit-image", "trimesh", "xatlas", "usd-core", "opencv-python-headless")}
        identity = fingerprint({"task": manifest, "code": code, "native": sha(native), "raster": sha(raster),
                                "modelCode": model_code, "weights": sha(runtime["weights"]),
                                "python": sys.version, "packages": packages})
        existing = root / "task.json"
        if existing.exists() and json.loads(existing.read_text()) != manifest:
            raise ValueError("Task input is immutable; create a new task directory")
        atomic_json(existing, manifest)
        atomic_json(root / "photos.json", photos)
        atomic_json(root / "input-use.json", {"allPhotos": all_photos, "photoUse": manifest.get("photoUse", [])})
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(Path(__file__).resolve().parent.parent),
                           REBUILD3D_RASTER_LIBRARY=str(raster), PYTHONUNBUFFERED="1",
                           PYTORCH_ENABLE_MPS_FALLBACK="1", HF_HUB_OFFLINE="1")
        python = [sys.executable, "-m"]
        stage = lambda name, *args: python + ["rebuild3d_worker." + name] + [str(a) for a in args]
        memory_gib = float(parameters.get("memoryGiB", 10.5))
        if not 2 <= memory_gib <= 11:
            raise ValueError("Invalid memory budget")
        stages = [
            ("prepare", "inputs", [str(native), "prepare", str(root / "photos.json"), str(root / "inputs")]),
            ("inference", "inference", stage("infer", root / "inputs", root / "inference", "--repo", runtime["vggtRepository"],
                "--weights", runtime["weights"], "--size", parameters.get("modelSize", 518), "--max-memory-gib", memory_gib)),
            ("fusion", "fusion", stage("fuse", root / "inference", root / "fusion", "--resolution", parameters.get("volumeResolution", 256))),
            ("unwrap", "uv-native", stage("unwrap", root / "fusion/face-provenance.npz", root / "uv-native", "--size", 2048)),
            ("atlas", "uv", stage("scale_uv", root / "uv-native", root / "uv", "--size", parameters.get("textureSize", 4096))),
            ("texture", "texture", stage("bake", root / "inputs", root, root / "uv", root / "texture",
                "--selection", "quality", "--quality-scale", 24, "--full-mask", "--safe-fill", "--smooth", 8)),
            ("package", "bundle", stage("package", root, root / "bundle")),
        ]
        prior, downstream_invalid = identity, False
        for name, directory, command in stages:
            self.check_cancelled()
            signature = fingerprint({"parent": prior, "command": command})
            checkpoint_path = root / "checkpoints" / (name + ".json")
            try:
                checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                checkpoint = {}
            if not downstream_invalid and checkpoint_valid(checkpoint, signature, root):
                self.emit("stage-reused", name)
            else:
                downstream_invalid = True
                output = root / directory
                if output.exists():
                    output.rename(root / "attempts" / (directory + "-" + str(uuid.uuid4())))
                stats = self.execute(name, command, environment, memory_gib * 1024**3)
                artifacts = artifact_manifest(output, root)
                if not artifacts:
                    raise RuntimeError("Stage produced no artifacts: " + name)
                checkpoint = {"signature": signature, "artifacts": artifacts, **stats}
                atomic_json(checkpoint_path, checkpoint)
                self.emit("stage-completed", name, **stats)
            prior = fingerprint(checkpoint)
        self.save_state(status="completed", stage="package", bundlePath=str(root / "bundle"), visualAcceptance="pending")
        self.emit("completed", "package", bundlePath=str(root / "bundle"), visualAcceptance="pending")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    supervisor = Supervisor(json.loads(args.manifest.read_text()), args.directory.resolve())
    lock = (supervisor.root / ".worker.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"protocolVersion": 1, "event": "failed", "message": "This task is already running"}), flush=True)
        sys.exit(2)
    existing = supervisor.root / "task.json"
    if existing.exists() and json.loads(existing.read_text()) != supervisor.manifest:
        print(json.dumps({"protocolVersion": 1, "event": "failed", "message": "Task input is immutable; use a new directory"}), flush=True)
        sys.exit(2)
    signal.signal(signal.SIGTERM, supervisor.stop)
    signal.signal(signal.SIGINT, supervisor.stop)
    try:
        supervisor.run()
    except Cancelled:
        supervisor.save_state(status="cancelled")
        supervisor.emit("cancelled", supervisor.state.get("stage"))
        sys.exit(130)
    except Exception as error:
        supervisor.save_state(status="failed", message=str(error))
        supervisor.emit("failed", supervisor.state.get("stage"), message=str(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
