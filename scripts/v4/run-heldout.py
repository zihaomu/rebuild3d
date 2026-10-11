#!/usr/bin/env python3
"""Freeze the candidate before first use of a registered E/E2; run fresh core tasks.

No tuning on E, no UI/performance claim. If E informs a later configuration
change, register a new independent held-out object before final validation.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

HELPER = Path(__file__).with_name('run-input-matrix.py')
SPEC = importlib.util.spec_from_file_location('matrix', HELPER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
H = M.H


def load(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('registry', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--baseline-runtime', type=Path, required=True)
    p.add_argument('--import-probe', type=Path, required=True)
    p.add_argument('--generation-probe', type=Path, required=True)
    p.add_argument('--phase-timeout', type=float, default=1800,
                   help='Maximum seconds per quality task, including first component preparation')
    a = p.parse_args()
    assert a.phase_timeout > 0
    registry, out = a.registry.resolve(), a.output.resolve()
    raw = registry.read_bytes()
    data = json.loads(raw)
    identifier = data['id']
    assert re.fullmatch(r'E(?:[1-9][0-9]*)?', identifier) and data['firstInferenceRun'] is None
    sources = [{'path': str((Path(data['directory']) / f['filename']).resolve()), 'sha256': f['sha256']}
               for f in data['files']]
    assert len(sources) == data['photoCount'] == 7
    assert all(H.sha(s['path']) == s['sha256'] for s in sources)
    runtime, baseline = a.runtime.resolve(), a.baseline_runtime.resolve()
    probes = [a.import_probe.resolve(), a.generation_probe.resolve()]
    for probe in probes:
        build = load(probe.parent / 'build.json')
        assert H.sha(probe) == build['binarySHA256']
        assert build.get('exitCode', build.get('observedCompileExitCode')) == 0
        for source, digest in build['sources'].items():
            assert H.sha(source) == digest, source
    out.mkdir(parents=True, exist_ok=False)
    for f in [Path(__file__), HELPER, M.HELPER, Path(__file__).with_name('audit-application-export.py')]:
        shutil.copy2(f, out / f.name)
    (out / 'registry-before.json').write_bytes(raw)
    H.save(out / 'inputs.json', {'sources': sources})
    for label, seed in [('baseline', baseline), ('candidate', runtime)]:
        shutil.copy2(seed / 'runtime.json', out / (label + '-runtime.json'))
    freeze = {'frozenAtUTC': H.now(), 'heldoutID': identifier,
              'candidateRuntimeManifestSHA256': H.sha(runtime / 'runtime.json'),
              'baselineRuntimeManifestSHA256': H.sha(baseline / 'runtime.json'),
              'registrySHA256BeforeFirstUse': H.sha(registry), 'scriptSHA256': H.sha(__file__),
              'runtime': str(runtime), 'baselineRuntime': str(baseline),
              'sources': sources, 'parameters': {'modelSize': 518, 'volumeResolution': 256, 'textureSize': 4096},
              'qualityProtocolSHA256': H.sha('build/v4-quantization/baseline/protocol.json'),
              'phaseTimeoutSeconds': a.phase_timeout,
              'policy': 'Configuration frozen before either E inference. Never select thresholds, precision, masks or geometry rules on E; a later E-informed change requires a new held-out object.'}
    H.save(out / 'configuration-freeze.json', freeze)
    report = {'status': 'running', 'scope': __doc__, 'freezeSHA256': H.sha(out / 'configuration-freeze.json'),
              'records': [], 'startedAtUTC': H.now()}
    H.save(out / 'report.json', report)
    try:
        seed_project = out / (identifier + '-imported.rebuild3d')
        command = [str(probes[0]), str(out / 'inputs.json'), str(seed_project)]
        with (out / 'import.log').open('w') as log:
            imported = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        H.save(out / 'import-process.json', {'command': command, 'exitCode': imported.returncode})
        assert imported.returncode == 0
        before = H.snapshot(seed_project)
        original_photos = load(seed_project / 'project.json')['photos']
        # Register before dispatch; a failed attempt is preserved rather than silently retried as a new first use.
        data['firstInferenceRun'] = {'output': str(out), 'registeredAtUTC': H.now(),
                                     'configurationFreezeSHA256': report['freezeSHA256'],
                                     'status': 'first-attempt-dispatched', 'note': 'Actual inference execution is proven by the stage events and reports.'}
        H.save(registry, data)
        for label, seed in [('baseline', baseline), ('candidate', runtime)]:
            base = out / label
            base.mkdir()
            project = base / (identifier + '.rebuild3d')
            subprocess.run(['/bin/cp', '-cR', str(seed_project), str(project)], check=True)
            record = {'name': label, 'project': str(project), 'status': 'running',
                      'runtimeManifestSHA256': H.sha(seed / 'runtime.json')}
            report['records'].append(record)
            H.save(out / 'report.json', report)
            result = H.run_phase(probes[1], seed, base / 'runtime-store', project, base / 'generated', None,
                                 timeout_seconds=a.phase_timeout)
            assert result['exitCode'] == 0, label
            photos = load(project / 'project.json')['photos']
            assert photos == original_photos
            jobs = list((project / 'generation').glob('*/request.json'))
            assert len(jobs) == 1
            job = jobs[0].parent
            task = load(jobs[0])
            assert task['parameters'] == {**freeze['parameters'], 'memoryGiB': 10.5}
            assert [r['id'] for r in task['photos']] == [r['id'] for r in photos]
            assert task['photos'] == task['allPhotos']
            events = [json.loads(line) for line in (job / 'events.jsonl').read_text().splitlines()]
            assert not any(e['event'] == 'stage-reused' for e in events)
            if label == 'candidate':
                record.update(M.validate_job(project, sources))
            command = [str(seed / 'python/bin/python3.12'), '-B',
                       str(Path(__file__).with_name('audit-application-export.py').resolve()),
                       str(project), str(base / 'generated/recovered.usdz'), str(base / 'export-audit')]
            with (base / 'export-audit.log').open('w') as log:
                audit = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'})
            H.save(base / 'export-process.json', {'command': command, 'exitCode': audit.returncode})
            assert audit.returncode == 0
            record.update(status='fresh-core-generation-and-export-passed', job=str(job),
                          photoCount=7, freshStageReuseCount=0,
                          exportAuditSHA256=H.sha(base / 'export-audit/report.json'))
            H.save(out / 'report.json', report)
            print(json.dumps({'label': label, 'status': record['status']}), flush=True)
        assert before == H.snapshot(seed_project)
        report['status'] = 'fresh-core-pair-passed-quality-pending'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['originalSourcesUnchanged'] = all(H.sha(s['path']) == s['sha256'] for s in sources)
        report['finishedAtUTC'] = H.now()
        H.save(out / 'report.json', report)
        if data['firstInferenceRun'] is not None:
            data['firstInferenceRun']['status'] = report['status']
            data['firstInferenceRun']['reportSHA256'] = H.sha(out / 'report.json')
            H.save(registry, data)
        assert report['originalSourcesUnchanged']


if __name__ == '__main__':
    main()
