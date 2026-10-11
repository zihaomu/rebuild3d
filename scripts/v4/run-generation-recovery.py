#!/usr/bin/env python3
"""Exercise the production Swift coordinator with real candidate workers.

Only copied projects and an isolated component store are modified. This is
core integration evidence, not actual UI, offline or performance acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import psutil


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def snapshot(root):
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.relative_to(root).parts[0] != "generation"}


def unchanged(before, after, *, except_manifest=False):
    return all(after.get(k) == v for k, v in before.items()
               if not (except_manifest and k == "project.json"))


def process_record(process):
    record = {"pid": process.pid}
    for name, read in [("created", process.create_time), ("command", process.cmdline),
                       ("rss", lambda: process.memory_info().rss)]:
        try:
            record[name] = read()
        except psutil.AccessDenied as error:
            # macOS can refuse KERN_PROCARGS2 around process exit. Missing
            # observations are recorded, never treated as zero or permission
            # to signal a process whose ownership has not been established.
            record[name] = None
            record[name + "Error"] = repr(error)
    return record


def surviving(records):
    result = []
    for item in records.values():
        try:
            p = psutil.Process(item["pid"])
            if (item["created"] is None or p.create_time() == item["created"]) and p.status() != psutil.STATUS_ZOMBIE:
                result.append(process_record(p))
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            result.append({**item, "terminationUnverified": repr(error)})
    return result


def run_phase(probe, seed, store, project, output, fault, *, timeout_seconds=600):
    if timeout_seconds <= 0:
        raise ValueError("Phase timeout must be positive")
    command = [str(probe), str(project), str(seed), str(store), str(output)]
    records, action, first_seen = {}, None, None
    observation_errors = {}
    started = time.monotonic()
    with output.with_suffix(".log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        observer = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                children = observer.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            except psutil.AccessDenied as error:
                children = []
                observation_errors[(child.pid, "children")] = {"atUTC": now(), "error": repr(error)}
            target = None
            for p in children:
                try:
                    item = process_record(p)
                except psutil.NoSuchProcess:
                    continue
                key = (item["pid"], item["created"])
                records[key] = item
                for name in ("created", "command", "rss"):
                    if name + "Error" in item:
                        observation_errors[(item["pid"], name)] = {"atUTC": now(), "record": item}
                module = "rebuild3d_worker.unwrap" if fault == "cancel-unwrap" else "rebuild3d_worker.infer"
                args = item["command"] or []
                if item["created"] is not None and len(args) > 2 and args[1:3] == ["-m", module] and str(project) in " ".join(args):
                    target = item
            if fault and target and not action:
                first_seen = first_seen or time.monotonic()
                delay = 0.75 if fault == "cancel-unwrap" else 2.0
                if time.monotonic() - first_seen >= delay:
                    action = {"kind": fault, "atUTC": now(), "target": target,
                              "observedLiveSecondsBeforeAction": time.monotonic() - first_seen}
                    if fault == "cancel-unwrap":
                        (output / "request-cancel").write_text(now() + "\n")
                    else:
                        # Revalidate ownership and PID identity immediately before the deliberate crash.
                        p = psutil.Process(target["pid"])
                        assert p.create_time() == target["created"]
                        assert child.pid in [parent.pid for parent in p.parents()]
                        p.send_signal(signal.SIGKILL)
                    save(output / "fault-action.json", action)
                    print(json.dumps({"phase": output.name, "action": action}, ensure_ascii=False), flush=True)
            if time.monotonic() - started > timeout_seconds:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise RuntimeError(f"Probe exceeded {timeout_seconds} seconds; retained logs and stopped only owned probe")
            time.sleep(0.1)
        code = child.wait()
    deadline = time.monotonic() + 15
    alive = surviving(records)
    while alive and time.monotonic() < deadline:
        time.sleep(0.25)
        alive = surviving(records)
    result = {"command": command, "exitCode": code, "elapsedSeconds": time.monotonic() - started,
              "faultAction": action, "observedDescendants": list(records.values()),
              "observationErrors": list(observation_errors.values()),
              "survivingDescendants": alive, "finishedAtUTC": now()}
    save(output / "process-result.json", result)
    assert not alive, "Owned stage processes survived the coordinator"
    if fault:
        assert action, "The intended interruption never happened"
    return result


def new_job(project, initial_jobs):
    jobs = {p for p in (project / "generation").iterdir() if p.is_dir()} - initial_jobs
    assert len(jobs) == 1, f"Expected exactly one candidate job, got {jobs}"
    return jobs.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--project-a", required=True, type=Path)
    parser.add_argument("--project-b", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=["A", "B"], default=["A", "B"])
    args = parser.parse_args()
    output, probe, seed = args.output.resolve(), args.probe.resolve(), args.runtime.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Unmodified production Swift coordinator, real packaged candidate; CLI only",
              "status": "running", "startedAtUTC": now(), "probeSHA256": sha(probe),
              "requestedDatasets": args.datasets,
              "runtimeManifestSHA256": sha(seed / "runtime.json"), "records": [],
              "sources": {str(p): sha(p) for p in sorted(Path("Sources/Rebuild3DCore").glob("*.swift"))},
              "harnessSHA256": sha(__file__),
              "swiftProbeSHA256": sha(Path(__file__).with_name("probe-generation-recovery.swift")),
              "limits": ["Not actual app UI or app-restart acceptance", "Not offline acceptance",
                         "Not controlled performance evidence", "Not an allocator/392 retry test"]}
    save(output / "report.json", report)
    try:
        for dataset, source, fault in [("A", args.project_a.resolve(), "cancel-unwrap"),
                                       ("B", args.project_b.resolve(), "kill-inference")]:
            if dataset not in args.datasets:
                continue
            before_source = snapshot(source)
            project = output / (dataset + "-recovery.rebuild3d")
            subprocess.run(["/bin/cp", "-cR", str(source), str(project)], check=True)
            before = snapshot(project)
            assert before == before_source
            save(output / (dataset + "-protected-before.json"), before)
            initial_jobs = {p for p in (project / "generation").iterdir() if p.is_dir()}
            first_output = output / (dataset + "-interrupted")
            first = run_phase(probe, seed, output / "runtime-store", project, first_output, fault)
            assert first["exitCode"] == (130 if fault == "cancel-unwrap" else 1)
            job = new_job(project, initial_jobs)
            state = json.loads((job / "state.json").read_text())
            expected = "cancelled" if fault == "cancel-unwrap" else "failed"
            assert state["status"] == expected, state
            request_sha = sha(job / "request.json")
            checkpoints = {p.name: sha(p) for p in (job / "checkpoints").glob("*.json")}
            expected_reused = ["prepare", "inference", "fusion"] if fault == "cancel-unwrap" else ["prepare"]
            assert set(checkpoints) == {name + ".json" for name in expected_reused}, checkpoints
            after_interruption = snapshot(project)
            save(output / (dataset + "-protected-after-interruption.json"), after_interruption)
            assert after_interruption == before, "Interruption changed published project data"
            old_events = (job / "events.jsonl").read_text().splitlines()
            record = {"dataset": dataset, "sourceProject": str(source), "copiedProject": str(project),
                      "job": str(job), "interruptionExitCode": first["exitCode"],
                      "interruptedState": state, "requestSHA256": request_sha,
                      "retainedCheckpoints": checkpoints, "oldProjectDataUnchangedAfterInterruption": True,
                      "protectedFiles": len(before), "status": "interruption-verified"}
            report["records"].append(record)
            save(output / "report.json", report)
            print(json.dumps({"dataset": dataset, "status": record["status"]}, ensure_ascii=False), flush=True)

            resumed_output = output / (dataset + "-resumed")
            resumed = run_phase(probe, seed, output / "runtime-store", project, resumed_output, None)
            assert resumed["exitCode"] == 0, "Coordinator resume did not complete; inspect phase report"
            assert new_job(project, initial_jobs) == job
            assert sha(job / "request.json") == request_sha
            events = [json.loads(line) for line in (job / "events.jsonl").read_text().splitlines()[len(old_events):]]
            reused = [e["stage"] for e in events if e["event"] == "stage-reused"]
            assert reused == expected_reused, reused
            assert all(sha(job / "checkpoints" / name) == digest for name, digest in checkpoints.items())
            assert json.loads((job / "state.json").read_text())["status"] == "completed"
            after_resume = snapshot(project)
            save(output / (dataset + "-protected-after-resume.json"), after_resume)
            assert unchanged(before, after_resume, except_manifest=True), "Recovery overwrote an old artifact"
            assert after_resume["project.json"] != before["project.json"], "New result was never committed"
            manifest = json.loads((project / "project.json").read_text())
            old_manifest = json.loads((source / "project.json").read_text())
            assert manifest["model"]["path"] != old_manifest["model"]["path"]
            assert manifest["photos"] == old_manifest["photos"]
            assert snapshot(source) == before_source, "Original acceptance project changed"
            record.update({"status": "core-recovery-passed", "resumedStages": reused,
                          "sameJobAndRequest": True, "retainedCheckpointsUnchanged": True,
                          "oldArtifactsRetainedAfterNewPublication": True,
                          "sourceProjectUnchanged": True, "photoRecordsUnchanged": True,
                          "newModelPath": manifest["model"]["path"], "resumeExitCode": 0})
            save(output / "report.json", report)
            print(json.dumps({"dataset": dataset, "status": record["status"]}, ensure_ascii=False), flush=True)
        report["status"] = "core-recovery-passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["finishedAtUTC"] = now()
        save(output / "report.json", report)


if __name__ == "__main__":
    main()
