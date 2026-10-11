#!/usr/bin/env python3
"""Freeze twelve fresh projects for serial interleaved real-UI measurements.

Creates imported projects only. It never launches the app or generates models.
Use observe-application-run.py before each actual Generate click and retain the
operator's click/display timestamp bounds. Quality runs must finish first.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline-app', type=Path, required=True)
    parser.add_argument('--candidate-app', type=Path, required=True)
    parser.add_argument('--import-probe', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path('build/v4-quantization/baseline/protocol.json'))
    args = parser.parse_args()
    out, importer = args.output.resolve(), args.import_probe.resolve()
    apps = {'baseline': args.baseline_app.resolve(), 'candidate': args.candidate_app.resolve()}
    build = load(importer.parent / 'build.json')
    assert sha(importer) == build['binarySHA256']
    assert all(sha(path) == digest for path, digest in build['sources'].items())
    protocol = load(args.protocol)
    identities = {}
    for label, app in apps.items():
        runtime = app / 'Contents/Resources/GenerationRuntime'
        manifest = runtime / 'runtime.json'
        for artifact in load(manifest)['artifacts']:
            path = runtime / artifact['path']
            assert path.resolve().is_relative_to(runtime)
            assert path.stat().st_size == artifact['byteCount'] and sha(path) == artifact['sha256']
        identities[label] = {'app': str(app), 'binarySHA256': sha(app / 'Contents/MacOS/Rebuild3D'),
                             'runtimeManifestSHA256': sha(manifest)}
    assert identities['baseline']['runtimeManifestSHA256'] == protocol['runtimeManifestSHA256']
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(args.protocol, out / 'quality-protocol.json')
    seeds = {}
    sources = {}
    for dataset in ('A', 'B'):
        original = load(args.protocol.parent / (dataset + '-task-template.json'))
        sources[dataset] = [{'path': photo['sourcePath'], 'sha256': photo['sourceSHA256']}
                            for photo in original['photos']]
        assert len(sources[dataset]) == 7
        assert all(sha(x['path']) == x['sha256'] for x in sources[dataset])
        inputs = out / (dataset + '-inputs.json')
        inputs.write_text(json.dumps({'sources': sources[dataset]}, indent=2) + '\n')
        seed = out / (dataset + '-imported-seed.rebuild3d')
        with (out / (dataset + '-import.log')).open('w') as log:
            subprocess.run([str(importer), str(inputs), str(seed)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = load(seed / 'project.json')
        assert not manifest.get('model') and not list((seed / 'generation').glob('*/request.json'))
        seeds[dataset] = seed
    order = []
    for round_index in range(3):
        for dataset in ('A', 'B'):
            labels = ['baseline', 'candidate']
            if (round_index + (dataset == 'B')) % 2:
                labels.reverse()
            for label in labels:
                name = f'{len(order)+1:02d}-{dataset}-{label}-r{round_index+1}'
                project = out / (name + '.rebuild3d')
                subprocess.run(['/bin/cp', '-cR', str(seeds[dataset]), str(project)], check=True)
                manifest = load(project / 'project.json')
                manifest.update(id=str(uuid.uuid4()).upper(), name=name)
                (project / 'project.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
                assert not manifest.get('model') and not list((project / 'generation').glob('*/request.json'))
                assert not any((project / 'cache').glob('**/*'))
                photos = {p['imagePath']: sha(project / p['imagePath']) for p in manifest['photos']}
                order.append({'name': name, 'dataset': dataset, 'label': label, 'round': round_index+1,
                              'project': str(project), 'projectManifestSHA256': sha(project / 'project.json'),
                              'photoFilesSHA256': photos, 'status': 'not-started'})
    report = {'status': 'prepared-not-started', 'scope': __doc__,
              'preparedAtUTC': datetime.now(timezone.utc).isoformat(), 'apps': identities,
              'scriptSHA256': sha(Path(__file__)), 'importProbeSHA256': sha(importer),
              'qualityProtocolSHA256': sha(args.protocol), 'sources': sources, 'order': order,
              'fixedParameters': {'photoCount': 7, 'modelSize': 518, 'volumeResolution': 256, 'textureSize': 4096},
              'rules': ['Finish owned quality/recovery/render/archive jobs before measurements',
                        'Restart the corresponding actual app for each sample; components already prepared',
                        'Open the fresh project, arm the observer, then click Generate through actual UI',
                        'Record click-call start/end and first verified model-visible screenshot time bounds',
                        'Use conservative waiting-time bounds; report observation uncertainty',
                        'Verify no stage reuse, no 392 retry, original hashes, exact app and runtime identity',
                        'Preserve failed/uncertain samples; do not mix parallel or recovery runs into medians',
                        'Judge A and B separately; keep 20% inference/memory and 15% UI waiting thresholds',
                        'Review power, thermal state, background processes, pressure and swap before passing'],
              'guiAcceptance': False, 'goalComplete': False}
    (out / 'protocol.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(out / 'protocol.json')


if __name__ == '__main__':
    main()
