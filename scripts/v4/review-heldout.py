#!/usr/bin/env python3
"""Native and numeric held-out review after the registered frozen fresh task pair.

Whole-photo regions are selected by image dimensions, never by model quality.
Visual acceptance remains manual and separate from numeric thresholds.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image


def sha(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()


def load(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--renderer', required=True, type=Path)
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--candidate-label', default='W8-candidate')
    a = p.parse_args()
    source, out = a.run.resolve(), a.output.resolve()
    run = load(source / 'report.json')
    assert run['status'] in ('fresh-core-pair-passed-quality-pending', 'core-pair-passed-quality-pending')
    assert run['freezeSHA256'] == sha(source / 'configuration-freeze.json')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    report = {'status': 'running', 'scope': __doc__, 'sourceRunSHA256': sha(source / 'report.json'),
              'configurationFreezeSHA256': run['freezeSHA256'], 'rendererSHA256': sha(a.renderer),
              'scriptSHA256': sha(Path(__file__)), 'processes': []}
    def save():
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    def execute(label, command, allowed=(0,)):
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report['processes'].append({'name': label, 'command': command, 'exitCode': result.returncode})
        save()
        assert result.returncode in allowed, label
    save()
    try:
        jobs = {}
        originals = {}
        scripts = Path(__file__).resolve().parent
        for record in run['records']:
            label, actual = record['name'], Path(record['job'])
            assert record['status'] in ('fresh-core-generation-and-export-passed', 'resumed-core-generation-and-export-passed')
            originals[label] = {str(f.relative_to(actual)): sha(f) for f in actual.rglob('*') if f.is_file()}
            job = out / label; job.mkdir(); jobs[label] = job
            for name in ['inputs', 'bundle', 'inference', 'fusion']:
                (job / name).symlink_to(actual / name, target_is_directory=True)
            (job / 'texture').mkdir()
            for f in (actual / 'texture').iterdir():
                if f.is_file() and f.name != 'uv-audit.json': (job / 'texture' / f.name).symlink_to(f)
            execute(label + '-uv-audit', [sys.executable, '-B', str(scripts.parent / 'texture/audit_uv.py'),
                    str(job / 'texture'), str(job / 'fusion/face-provenance.npz')])
            execute(label + '-render', [str(a.renderer.resolve()), str(job / 'bundle/model.usdz'),
                    str(job / 'renders'), str(job / 'bundle/cameras.json'), str(job / 'bundle/fusion.json')])
            manifest = {'renderer': 'SceneKit native GPU offscreen; not application UI',
                        'rendererSHA256': report['rendererSHA256'], 'modelSHA256': sha(job / 'bundle/model.usdz'),
                        'images': {f.name: sha(f) for f in (job / 'renders').iterdir() if f.is_file()}}
            (job / 'renders/render-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        regions = []
        for i, record in enumerate(load(jobs['candidate'] / 'inputs/dataset.json')['records']):
            width, height = Image.open(jobs['candidate'] / 'inputs' / record['image']).size
            regions.append({'id': f'full-photo-{i}', 'photoSHA256': record['sourceSHA256'], 'box': [0, 0, width, height]})
        (out / 'regions.json').write_text(json.dumps({'selection': 'Every entire prepared image; no output-based region selection', 'regions': regions}, indent=2) + '\n')
        execute('inference-compare', [sys.executable, '-B', str(scripts / 'compare-inference.py'),
                str(jobs['baseline'] / 'inference'), str(jobs['candidate'] / 'inference'),
                str(out / 'inference-comparison.json'), '--protocol', str(a.protocol.resolve()), '--relocated-inputs'], (0, 1))
        inference = load(out / 'inference-comparison.json')
        execute('model-compare', [sys.executable, '-B', str(scripts / 'review-model.py'),
                str(jobs['baseline']), str(jobs['candidate']), str(out / 'regions.json'),
                '--protocol', str(a.protocol.resolve()), '--candidate-label', a.candidate_label], (0, 1))
        metrics = load(jobs['candidate'] / 'review-v4/metrics.json')
        for record in run['records']:
            actual = Path(record['job'])
            assert originals[record['name']] == {str(f.relative_to(actual)): sha(f) for f in actual.rglob('*') if f.is_file()}
        report.update(status='review-generated-visual-verdict-pending', originalsUnchanged=True,
                      inferenceNumericGatesPassed=inference['numericGatesPassed'], modelNumericGatesPassed=metrics['numericGatesPassed'])
    except BaseException as e:
        report.update(status='failed', error=repr(e)); raise
    finally:
        report['finishedAtUTC'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    main()
