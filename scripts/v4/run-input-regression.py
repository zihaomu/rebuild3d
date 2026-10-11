#!/usr/bin/env python3
"""Run registered D12/C33/C59 projects through the production coordinator.

Copies existing projects, preserves their originals and old results, and
requires fresh work for the candidate. CLI evidence does not prove UI behavior.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


HELPER = Path(__file__).with_name("run-generation-recovery.py")
SPEC = importlib.util.spec_from_file_location("recovery_harness", HELPER)
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--runtime-store", required=True, type=Path)
    parser.add_argument("--old-projects", type=Path, default=Path("build/v3-oneclick/delivery"))
    args = parser.parse_args()
    output, probe, seed, store = (p.resolve() for p in
                                  (args.output, args.probe, args.runtime, args.runtime_store))
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Production coordinator CLI, independent copies of registered v3 projects",
              "status": "running", "startedAtUTC": H.now(), "records": [],
              "scriptSHA256": H.sha(__file__), "helperSHA256": H.sha(HELPER),
              "probeSHA256": H.sha(probe), "runtimeManifestSHA256": H.sha(seed / "runtime.json"),
              "limits": ["Not application UI, offline, or controlled performance acceptance",
                         "New outputs still require artifact and visual review",
                         "Does not exercise additional 3-11 image or orientation/order variants",
                         "E held-out object is not included"]}
    H.save(output / "report.json", report)
    try:
        for name, original, count, expected_route in [
                ("D12", "D-skull-v3-auto", 12, "sparse"),
                ("C33", "C33-jpeg-v3", 33, "Object Capture"),
                ("C59", "C59-heic-v3", 59, "Object Capture")]:
            source = (args.old_projects / (original + ".rebuild3d")).resolve()
            before = H.snapshot(source)
            old = json.loads((source / "project.json").read_text())
            assert len(old["photos"]) == count
            for photo in old["photos"]:
                assert H.sha(source / photo["imagePath"]) == photo["metadata"]["contentSHA256"]
            if name == "D12":
                registered = json.loads(Path("build/v3-oneclick/heldout/skull-inputs.json").read_text())
                assert [(p["originalName"], p["metadata"]["contentSHA256"]) for p in old["photos"]] == [
                    (p["localName"], p["sha256"]) for p in registered]
            project = output / (name + ".rebuild3d")
            subprocess.run(["/bin/cp", "-cR", str(source), str(project)], check=True)
            assert H.snapshot(project) == before
            H.save(output / (name + "-protected-before.json"), before)
            initial_jobs = set((project / "generation").glob("*/request.json"))
            record = {"dataset": name, "photoCount": count, "source": str(source), "project": str(project),
                      "expectedRoute": expected_route, "originalManifestSHA256": H.sha(source / "project.json"),
                      "status": "running"}
            report["records"].append(record)
            H.save(output / "report.json", report)
            phase = output / (name + "-generated")
            result = H.run_phase(probe, seed, store, project, phase, None)
            record["exitCode"] = result["exitCode"]
            assert result["exitCode"] == 0, f"{name} generation failed; inspect {phase}"
            events = [json.loads(line) for line in (phase / "events.jsonl").read_text().splitlines()]
            routes = [e["message"] for e in events if "Automatic route:" in e.get("message", "")]
            assert routes and all("Automatic route: " + expected_route in r for r in routes), routes
            current = json.loads((project / "project.json").read_text())
            assert current["photos"] == old["photos"]
            assert current["model"]["path"] != old["model"]["path"]
            assert current["model"]["photoIDs"] == [p["id"] for p in current["photos"]]
            assert H.unchanged(before, H.snapshot(project), except_manifest=True), "Old artifact overwritten"
            assert H.snapshot(source) == before, "Original project modified"
            new_jobs = set((project / "generation").glob("*/request.json")) - initial_jobs
            if name == "D12":
                assert len(new_jobs) == 1, new_jobs
                job = next(iter(new_jobs)).parent
                task = json.loads((job / "request.json").read_text())
                assert len(task["photos"]) == len(task["allPhotos"]) == 12
                assert task["parameters"]["modelSize"] == 518
                worker_events = [json.loads(line) for line in (job / "events.jsonl").read_text().splitlines()]
                assert not any(e["event"] == "stage-reused" for e in worker_events)
                identity = json.loads((job / "runtime-identity.json").read_text())
                assert identity["backendID"] == "mlx-aggregator-pytorch-heads-v1"
                record.update(job=str(job), reusedStageCount=0, backendID=identity["backendID"],
                              runtimeIdentitySHA256=H.sha(job / "runtime-identity.json"))
                assert current["model"].get("approximation")
            else:
                assert not new_jobs, "Conventional regression silently fell back to sparse"
                assert not current["model"].get("approximation")
                logs = [json.loads(p.read_text()) for p in (project / "logs").glob("*.json")]
                run_report = next(r for r in logs if r["id"] == current["model"]["runID"])
                assert run_report["status"] == "completed" and run_report["photoCount"] == count
                record["conventionalRunReport"] = run_report
            record.update(status="core-generation-and-export-passed", actualRoutes=routes,
                          photoIDsPreserved=True, sourceAndOldArtifactsUnchanged=True,
                          outputModelSHA256=H.sha(project / current["model"]["path"]),
                          exportedUSDZ=str(phase / "recovered.usdz"),
                          verificationReportSHA256=H.sha(phase / "report.json"))
            H.save(output / "report.json", report)
            print(json.dumps({"dataset": name, "status": record["status"], "routes": routes}), flush=True)
        report["status"] = "core-input-routes-passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["finishedAtUTC"] = H.now()
        H.save(output / "report.json", report)


if __name__ == "__main__":
    main()
