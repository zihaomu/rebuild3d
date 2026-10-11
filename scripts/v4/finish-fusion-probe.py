#!/usr/bin/env python3
"""Bake and natively render isolated fusion variants without changing inputs.

Reuses prepared inputs, inference and fusion, so this is a quality experiment,
not a fresh task, application acceptance, or end-to-end performance sample.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_job', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--worker-root', type=Path, required=True)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--uv-auditor', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'texture/audit_uv.py')
    parser.add_argument('--variant', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'INFERENCE', 'FUSION'))
    args = parser.parse_args()
    source, out, worker, renderer = (p.resolve() for p in
        (args.source_job, args.output, args.worker_root, args.renderer))
    uv_auditor = args.uv_auditor.resolve()
    labels = [v[0] for v in args.variant]
    if len(set(labels)) != len(labels) or any(Path(v).name != v or v in ('.', '..') for v in labels):
        raise ValueError('Variant labels must be unique simple directory names')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(worker), PYTHONDONTWRITEBYTECODE='1',
        REBUILD3D_RASTER_LIBRARY=str(worker / 'rebuild3d_worker/raster.dylib'),
        REBUILD3D_UV_LAYOUT_LIBRARY=str(worker / 'rebuild3d_worker/uv_layout.dylib'),
        PYTHONUNBUFFERED='1', PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1')
    original_inputs = files(source / 'inputs')
    photo_hashes = [r['sourceSHA256'] for r in
                    json.loads((source / 'inputs/dataset.json').read_text())['records']]
    report = {'status': 'running', 'startedAtUTC': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__, 'sourceJob': str(source), 'orderedSourceSHA256': photo_hashes,
              'scriptSHA256': sha(Path(__file__)), 'workerFiles': files(worker),
              'uvAuditorSHA256': sha(uv_auditor),
              'uvAuditorCommonSHA256': sha(uv_auditor.with_name('common.py')),
              'rendererSHA256': sha(renderer), 'inputFiles': original_inputs, 'variants': []}
    save(out / 'report.json', report)
    try:
        for label, inf, fused in args.variant:
            inference, fusion = Path(inf).resolve(), Path(fused).resolve()
            config = json.loads((inference / 'config.json').read_text())
            assert config['sourceSHA256'] == photo_hashes, 'Ordered source identities differ'
            records = json.loads((source / 'inputs/dataset.json').read_text())['records']
            transforms = json.loads((inference / 'pixel-transforms.json').read_text())
            assert [r['id'] for r in records] == [t['photoID'] for t in transforms], \
                'Prepared photo IDs differ; use the inference task original inputs'
            before = {'inference': files(inference), 'fusion': files(fusion)}
            job = out / label
            job.mkdir()
            for name, target in [('inputs', source / 'inputs'), ('inference', inference), ('fusion', fusion)]:
                (job / name).symlink_to(target, target_is_directory=True)
            shutil.copy2(source / 'input-use.json', job / 'input-use.json')
            item = {'label': label, 'job': str(job), 'reused': before, 'stages': [],
                    'status': 'running', 'inferenceConfigSHA256': sha(inference / 'config.json')}
            report['variants'].append(item)
            module = lambda name, *params: [sys.executable, '-B', '-m',
                                           'rebuild3d_worker.' + name, *map(str, params)]
            stages = [
                ('unwrap', module('unwrap', fusion / 'face-provenance.npz', job / 'uv-native', '--size', 2048)),
                ('atlas', module('scale_uv', job / 'uv-native', job / 'uv', '--size', 4096,
                                 '--geometry', fusion / 'face-provenance.npz')),
                ('texture', module('bake', job / 'inputs', job, job / 'uv', job / 'texture',
                                   '--selection', 'quality', '--quality-scale', 24,
                                   '--full-mask', '--safe-fill', '--smooth', 8)),
                ('uv-audit', [sys.executable, '-B', str(uv_auditor), str(job / 'texture'),
                              str(fusion / 'face-provenance.npz')]),
                ('package', module('package', job, job / 'bundle')),
                ('render', [str(renderer), str(job / 'bundle/model.usdz'), str(job / 'renders'),
                            str(job / 'bundle/cameras.json'), str(job / 'bundle/fusion.json')]),
            ]
            for name, command in stages:
                started = time.monotonic()
                with (job / (name + '.log')).open('w') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                stage = {'name': name, 'command': command, 'exitCode': result.returncode,
                         'diagnosticElapsedSeconds': time.monotonic() - started}
                item['stages'].append(stage)
                save(out / 'report.json', report)
                print(json.dumps({'variant': label, **stage}), flush=True)
                if result.returncode:
                    raise RuntimeError(f'{label} {name} exited {result.returncode}')
            assert before == {'inference': files(inference), 'fusion': files(fusion)}
            for i in range(len(photo_hashes)):
                for mode in ('color', 'clay', 'mask', 'lit'):
                    assert (job / 'renders' / f'view-{i}-{mode}.png').is_file()
            for i in range(12):
                for mode in ('color', 'clay'):
                    assert (job / 'renders' / f'turntable-{mode}-{i:02d}.png').is_file()
            render_record = {'renderer': 'SceneKit native GPU offscreen; not application UI',
                             'rendererSHA256': sha(renderer), 'modelSHA256': sha(job / 'bundle/model.usdz'),
                             'images': files(job / 'renders')}
            save(job / 'renders/render-manifest.json', render_record)
            item.update(status='baked-and-rendered-visual-review-pending',
                        reusedInputsUnchanged=True, modelSHA256=render_record['modelSHA256'],
                        nativeRenderManifestSHA256=sha(job / 'renders/render-manifest.json'))
            save(out / 'report.json', report)
        assert files(source / 'inputs') == original_inputs
        report.update(status='baked-and-rendered-visual-review-pending', originalInputsUnchanged=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finishedAtUTC'] = datetime.now(timezone.utc).isoformat()
        save(out / 'report.json', report)


if __name__ == '__main__':
    main()
