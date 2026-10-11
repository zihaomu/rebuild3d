#!/usr/bin/env python3
"""Render completed core input cases and assemble all-view review sheets.

Rendering and artifact hashes do not constitute a visual verdict.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def tile(path, width=300, height=205):
    with Image.open(path) as image:
        image = ImageOps.contain(image.convert('RGB'), (width, height))
    canvas = Image.new('RGB', (width, height), '#202020')
    canvas.paste(image, ((width-image.width)//2, (height-image.height)//2))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_report', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--renderer', required=True, type=Path)
    parser.add_argument('--names', nargs='+', help='Explicit completed cases from an ongoing matrix; no whole-matrix claim')
    args = parser.parse_args()
    run_bytes = args.run_report.read_bytes()
    run = json.loads(run_bytes)
    if args.names:
        assert run['status'] in ('running', 'core-input-matrix-passed',
                                 'core-pair-passed-quality-pending',
                                 'fresh-core-pair-passed-quality-pending')
    else:
        assert run['status'] in ('core-input-matrix-passed', 'core-input-variant-passed',
                                 'controlled-allocator-core-retry-passed')
    records = run.get('records') or [{**run, 'name': args.run_report.parent.name}]
    if args.names:
        records = [r for r in records if r['name'] in args.names]
        assert {r['name'] for r in records} == set(args.names)
        assert all(r['status'] in ('core-input-and-export-passed',
                                  'fresh-core-generation-and-export-passed',
                                  'resumed-core-generation-and-export-passed') for r in records)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    (output / 'source-run-report.json').write_bytes(run_bytes)
    report = {'status': 'running', 'startedAtUTC': datetime.now(timezone.utc).isoformat(),
              'runReport': str(args.run_report.resolve()), 'runReportSHA256': hashlib.sha256(run_bytes).hexdigest(),
              'requestedCases': [r['name'] for r in records], 'explicitSubset': bool(args.names),
              'rendererSHA256': sha(args.renderer), 'scriptSHA256': sha(Path(__file__)), 'records': [],
              'scope': 'Native SceneKit renders of actual fresh outputs; visual decision must be recorded separately'}
    save(output / 'report.json', report)
    font = ImageFont.load_default(size=18)
    try:
        for record in records:
            job = Path(record['job'])
            name = record['name']
            base = output / name
            base.mkdir()
            rendered = base / 'native'
            command = [str(args.renderer.resolve()), str(job / 'bundle/model.usdz'), str(rendered),
                       str(job / 'bundle/cameras.json'), str(job / 'bundle/fusion.json')]
            with (base / 'render.log').open('w') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            save(base / 'process-result.json', {'command': command, 'exitCode': process.returncode})
            assert process.returncode == 0, name
            inputs = json.loads((job / 'inputs/dataset.json').read_text())['records']
            assert len(inputs) == record['photoCount']
            for i in range(len(inputs)):
                for mode in ('color', 'clay', 'mask', 'lit'):
                    assert (rendered / f'view-{i}-{mode}.png').is_file()
            rows = len(inputs)
            sheet = Image.new('RGB', (920, 45 + 235 * rows), '#101010')
            draw = ImageDraw.Draw(sheet)
            draw.text((10, 10), name + ' | input photograph / textured output / untextured output', fill='white', font=font)
            for i, source in enumerate(inputs):
                y = 45 + 235 * i
                draw.text((10, y), f'Photo {i + 1} | EXIF {source["encodedOrientation"]}', fill='white', font=font)
                for j, path in enumerate((job / 'inputs' / source['fullImage'],
                                          rendered / f'view-{i}-color.png', rendered / f'view-{i}-clay.png')):
                    sheet.paste(tile(path), (10 + 305 * j, y + 25))
            all_views = base / 'all-views.jpg'
            sheet.save(all_views, quality=95)
            turntable = Image.new('RGB', (1225, 45 + 450 * 3), '#101010')
            draw = ImageDraw.Draw(turntable)
            draw.text((10, 10), name + ' | 12 angles, color above clay', fill='white', font=font)
            for i in range(12):
                x, y = 10 + (i % 4) * 305, 45 + (i // 4) * 450
                draw.text((x, y), f'Angle {i}', fill='white', font=font)
                for j, mode in enumerate(('color', 'clay')):
                    path = rendered / f'turntable-{mode}-{i:02d}.png'
                    assert path.is_file(), path
                    turntable.paste(tile(path), (x, y + 25 + j * 210))
            turn_path = base / 'turntable.jpg'
            turntable.save(turn_path, quality=95)
            item = {'name': name, 'photoCount': len(inputs), 'job': str(job),
                    'modelSHA256': sha(job / 'bundle/model.usdz'),
                    'nativeImages': {p.name: sha(p) for p in sorted(rendered.glob('*.png'))},
                    'sheets': {p.name: sha(p) for p in (all_views, turn_path)}, 'visualVerdict': 'pending'}
            report['records'].append(item)
            save(output / 'report.json', report)
            print(json.dumps({'name': name, 'rendered': len(item['nativeImages']), 'visualVerdict': 'pending'}), flush=True)
        report['status'] = 'rendered-visual-review-pending'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finishedAtUTC'] = datetime.now(timezone.utc).isoformat()
        save(output / 'report.json', report)


if __name__ == '__main__':
    main()
