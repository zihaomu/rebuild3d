#!/usr/bin/env python3
"""Run one explicitly registered input variant through production import and generation."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

MATRIX = Path(__file__).with_name('run-input-matrix.py')
SPEC = importlib.util.spec_from_file_location('input_matrix', MATRIX)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
H = M.H


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_list', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--import-probe', required=True, type=Path)
    parser.add_argument('--generation-probe', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--runtime-store', required=True, type=Path)
    args = parser.parse_args()
    inputs = json.loads(args.input_list.read_text())
    sources = inputs['sources']
    assert 3 <= len(sources) <= 12
    output, runtime = args.output.resolve(), args.runtime.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for script in (Path(__file__), MATRIX, M.HELPER, Path(__file__).with_name('audit-application-export.py')):
        shutil.copy2(script, output / script.name)
    shutil.copy2(args.input_list, output / 'inputs.json')
    report = {'status': 'running', 'startedAtUTC': H.now(), 'inputListSHA256': H.sha(args.input_list),
              'runtimeManifestSHA256': H.sha(runtime / 'runtime.json'),
              'scriptSHA256': H.sha(__file__), 'validationScriptSHA256': H.sha(MATRIX),
              'importProbeSHA256': H.sha(args.import_probe), 'generationProbeSHA256': H.sha(args.generation_probe),
              'scope': 'Production import, fresh coordinator generation and persisted coordinates; no UI/offline/performance claim'}
    H.save(output / 'report.json', report)
    try:
        for source in sources:
            assert H.sha(source['path']) == source['sha256']
            if 'sourcePath' in source:
                assert H.sha(source['sourcePath']) == source['sourceSHA256']
        fixture_pixels = []
        if inputs.get('pixelComparisonThresholds'):
            thresholds = inputs['pixelComparisonThresholds']
            for source in sources:
                for key in ('uprightReference', 'nativeDecoded'):
                    assert H.sha(source[key]) == source[key + 'SHA256']
                with Image.open(source['uprightReference']) as image:
                    reference = np.asarray(image.convert('RGB'), dtype=float)
                with Image.open(source['nativeDecoded']) as image:
                    decoded = np.asarray(image.convert('RGB'), dtype=float)
                assert decoded.shape == reference.shape
                delta = np.abs(decoded - reference)
                row = {'orientation': source['orientation'], 'meanAbsoluteRGBError': float(delta.mean()),
                       'p99AbsoluteRGBError': float(np.percentile(delta, 99)), 'shape': list(decoded.shape)}
                assert row['meanAbsoluteRGBError'] <= thresholds['meanAbsoluteRGBError']
                assert row['p99AbsoluteRGBError'] <= thresholds['p99AbsoluteRGBError']
                fixture_pixels.append(row)
            report['fixturePixelComparison'] = fixture_pixels
            H.save(output / 'report.json', report)
        project = output / '方向与画幅 新项目.rebuild3d'
        command = [str(args.import_probe.resolve()), str((output / 'inputs.json').resolve()), str(project)]
        with (output / 'import.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        H.save(output / 'import-process.json', {'command': command, 'exitCode': result.returncode})
        assert result.returncode == 0
        photos = json.loads((project / 'project.json').read_text())['photos']
        if all('orientation' in source for source in sources):
            assert [p['orientation'] for p in photos] == [source['orientation'] for source in sources]
        for photo, source in zip(photos, sources):
            if 'expectedTypeIdentifier' in source:
                assert photo['metadata']['typeIdentifier'] == source['expectedTypeIdentifier']
                if source['expectedTypeIdentifier'] == 'public.heic':
                    assert Path(photo['imagePath']).suffix == '.heic'
        phase = output / 'generated'
        result = H.run_phase(args.generation_probe.resolve(), runtime, args.runtime_store.resolve(), project, phase, None)
        assert result['exitCode'] == 0
        report.update(M.validate_job(project, sources))
        assert json.loads((project / 'project.json').read_text())['photos'] == photos
        job = Path(report['job'])
        prepared = json.loads((job / 'inputs/dataset.json').read_text())['records']
        exact = []
        for source, record in zip(sources, prepared):
            if 'nativeDecoded' in source:
                with Image.open(source['nativeDecoded']) as image:
                    expected = np.asarray(image.convert('RGB'))
                with Image.open(job / 'inputs' / record['fullImage']) as image:
                    actual = np.asarray(image.convert('RGB'))
                assert np.array_equal(actual, expected), 'Prepared full image differs from verified upright fixture'
                exact.append({'photoID': record['id'], 'orientation': source['orientation'],
                              'fullImageSHA256': H.sha(job / 'inputs' / record['fullImage']), 'pixelsExactlyEqual': True})
        report['preparedOrientationProof'] = exact
        command = [str(runtime / 'python/bin/python3.12'), '-B', str(Path(__file__).with_name('audit-application-export.py').resolve()),
                   str(project), str(phase / 'recovered.usdz'), str(output / 'export-audit')]
        with (output / 'export-audit.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'})
        H.save(output / 'export-process.json', {'command': command, 'exitCode': result.returncode})
        assert result.returncode == 0
        report.update(status='core-input-variant-passed', project=str(project),
                      exportAuditSHA256=H.sha(output / 'export-audit/report.json'))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['sourcesUnchanged'] = all(H.sha(source['path']) == source['sha256'] for source in sources)
        report['finishedAtUTC'] = H.now()
        H.save(output / 'report.json', report)
        assert report['sourcesUnchanged']
    print(json.dumps({'status': report['status'], 'photoCount': report['photoCount'],
                      'orientations': report['orientations'], 'projectionError': report['maxProjectionChainErrorPixels']}))


if __name__ == '__main__':
    main()
