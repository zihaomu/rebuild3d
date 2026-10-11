#!/usr/bin/env python3
"""Prepare native D12/C33/C59 comparisons without mutating old projects.

D uses all recorded cameras and frozen details. Conventional results use a
12-angle turntable; no photographic camera alignment is claimed for C.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sheet(before, after, mode, target):
    canvas = Image.new('RGB', (1280, 2070), '#222222')
    draw = ImageDraw.Draw(canvas)
    for index in range(12):
        for row, (root, label) in enumerate(((before, 'v3'), (after, 'candidate05'))):
            image = Image.open(root / f'turntable-{mode}-{index:02d}.png').convert('RGB')
            image.thumbnail((316, 316), Image.Resampling.LANCZOS)
            x, y = index % 4 * 320, (index // 4 * 2 + row) * 345
            draw.text((x + 3, y + 3), f'{label} angle {index}', fill='white')
            canvas.paste(image, (x + 2, y + 26))
    canvas.save(target, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--renderer', required=True, type=Path)
    args = parser.parse_args()
    run, output, runtime, renderer = (p.resolve() for p in
                                       (args.run, args.output, args.runtime, args.renderer))
    source = read(run / 'report.json')
    assert source['status'] == 'core-input-routes-passed'
    output.mkdir(parents=True, exist_ok=False)
    python = runtime / 'python/bin/python3.12'
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(runtime / 'worker'), PYTHONDONTWRITEBYTECODE='1')
    summary = {'status': 'running', 'scriptSHA256': sha(Path(__file__)),
               'sourceRunReportSHA256': sha(run / 'report.json'), 'rendererSHA256': sha(renderer),
               'runtimeManifestSHA256': sha(runtime / 'runtime.json'), 'processes': [], 'datasets': [],
               'scope': 'Native offscreen renders and numerical comparison; visual judgment and actual UI are separate'}

    def save():
        (output / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    def execute(name, command, acceptable=(0,)):
        started = time.monotonic()
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run([str(x) for x in command], env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        summary['processes'].append({'name': name, 'command': [str(x) for x in command],
                                     'exitCode': result.returncode, 'elapsedSeconds': time.monotonic() - started})
        save()
        print(name, 'exit', result.returncode, flush=True)
        if result.returncode not in acceptable:
            raise RuntimeError(name + ' failed; retained log')

    save()
    try:
        d = next(record for record in source['records'] if record['dataset'] == 'D12')
        original_project, candidate_job = Path(d['source']), Path(d['job'])
        manifest = read(original_project / 'project.json')
        original_sha = sha(original_project / manifest['model']['path'])
        matches = [p.parent for p in (original_project / 'generation').glob('*/bundle/model.usdz')
                   if sha(p) == original_sha]
        assert len(matches) == 1
        original_job = matches[0].parent
        d_record = {'dataset': 'D12', 'baselineProjectModelSHA256': original_sha,
                    'baselineJob': str(original_job), 'candidateJob': str(candidate_job),
                    'baselineRenderPolicy': 'Re-export original geometry/UV/photo texture with current normal-preserving exporter; no geometry, UV or texture recomputation',
                    'copies': [], 'visualAcceptance': 'pending'}
        summary['datasets'].append(d_record)
        wrappers = []
        for label, job in (('baseline', original_job), ('candidate', candidate_job)):
            wrapper = output / 'D12' / label
            bundle = wrapper / 'bundle'
            bundle.mkdir(parents=True)
            for directory in ('inputs', 'fusion', 'inference'):
                (wrapper / directory).symlink_to(job / directory, target_is_directory=True)
            (wrapper / 'texture').symlink_to(bundle, target_is_directory=True)
            names = ['geometry.npz', 'uv.npz', 'basecolor.png', 'fusion.json', 'cameras.json',
                     'face-selection.npz', 'atlas-surface.npz', 'report.json',
                     'texture-source.png', 'texture-quality.png']
            if label == 'candidate':
                names += ['model.usdz', 'model.glb']
            for name in names:
                source_file = job / 'bundle' / name
                shutil.copy2(source_file, bundle / name)
                assert sha(source_file) == sha(bundle / name)
                d_record['copies'].append({'side': label, 'name': name, 'sha256': sha(source_file)})
            if label == 'baseline':
                execute('D12-baseline-normal-preserving-export', [python, '-B', '-m', 'rebuild3d_worker.export', bundle])
            execute('D12-' + label + '-uv-audit', [python, '-B', 'scripts/texture/audit_uv.py',
                                                   bundle, job / 'fusion/face-provenance.npz'])
            execute('D12-' + label + '-native-render', [renderer, bundle / 'model.usdz', wrapper / 'renders',
                                                       bundle / 'cameras.json', job / 'fusion/report.json'])
            wrappers.append(wrapper)
        execute('D12-model-comparison', [python, '-B', 'scripts/v4/review-model.py', *wrappers,
                                         'build/v3-oneclick/heldout/D-review-regions.json', '--protocol',
                                         'build/v4-quantization/baseline/protocol.json'], acceptable=(0, 1))
        d_record['metrics'] = str(wrappers[1] / 'review-v4/metrics.json')
        d_record['numericGatesPassed'] = read(Path(d_record['metrics']))['numericGatesPassed']
        cameras = output / 'conventional-no-photo-cameras.json'
        cameras.write_text(json.dumps({'cameras': [], 'scope': 'No photo-camera estimates supplied; turntable only'}) + '\n')
        transform = output / 'conventional-turntable-transform.json'
        transform.write_text(json.dumps({'worldToDisplay': [[1, 0, 0, 0], [0, 1, 0, 0],
                                                           [0, 0, 1, 0], [0, 0, 0, 1]]}) + '\n')
        for dataset in ('C33', 'C59'):
            record = next(r for r in source['records'] if r['dataset'] == dataset)
            old = Path(record['source'])
            model = old / read(old / 'project.json')['model']['path']
            outputs = []
            for label, model_path in (('baseline', model), ('candidate', Path(record['exportedUSDZ']))):
                destination = output / dataset / label
                destination.parent.mkdir(parents=True, exist_ok=True)
                execute(dataset + '-' + label + '-native-render', [renderer, model_path, destination, cameras, transform])
                outputs.append(destination)
            for mode in ('color', 'clay', 'lit'):
                sheet(*outputs, mode, output / dataset / (mode + '-turntable.jpg'))
            summary['datasets'].append({'dataset': dataset, 'baselineModelSHA256': sha(model),
                                         'candidateModelSHA256': sha(Path(record['exportedUSDZ'])),
                                         'cameraPolicy': 'Same bounding-box-relative 12 angles, not photo-camera correspondence',
                                         'visualAcceptance': 'pending'})
        summary['status'] = 'native-review-artifacts-ready'
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
