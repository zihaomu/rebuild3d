#!/usr/bin/env python3
"""Render labeled fusion experiments through their own cameras for diagnosis.

Each --variant contains a label, inference directory and fusion directory.
Original photos/order must match. This is a CPU geometry diagnostic, not native
texture, measured geometry, or application acceptance.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--variant', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'INFERENCE', 'FUSION'))
    args = parser.parse_args()
    helper = Path(__file__).with_name('diagnose-low-resolution-fusion.py')
    spec = importlib.util.spec_from_file_location('fusion_diagnostic', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    loaded, variants, sources = [], [], None
    for name, inference, fusion in args.variant:
        inference, fusion = Path(inference).resolve(), Path(fusion).resolve()
        config = json.loads((inference / 'config.json').read_text())
        if sources is None:
            sources = config['sourceSHA256']
        assert sources == config['sourceSHA256']
        with np.load(inference / 'predictions.npz') as data, \
                np.load(fusion / 'face-provenance.npz') as mesh:
            loaded.append(({k: data[k] for k in data.files},
                           mesh['verticesBeforeDisplay'], mesh['faces']))
        variants.append({'label': name, 'inference': str(inference), 'fusion': str(fusion),
                         'configSHA256': module.sha(inference / 'config.json'),
                         'predictionsSHA256': module.sha(inference / 'predictions.npz'),
                         'geometrySHA256': module.sha(fusion / 'face-provenance.npz'),
                         'fusionReportSHA256': module.sha(fusion / 'report.json')})
    count, cols = len(sources), len(variants) + 1
    sheet = Image.new('RGB', (cols * 280, 40 + count * 300), '#141414')
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(['Model input'] + [v['label'] for v in variants]):
        draw.text((col * 280 + 5, 10), label, fill='white')
    rows, images = [], {}
    for index in range(count):
        cells = [Image.fromarray(loaded[0][0]['rgb'][index])]
        scores = []
        for vindex, (data, vertices, faces) in enumerate(loaded):
            size = data['depth'].shape[-1]
            rendered, mask = module.draw_mesh(vertices, faces, data['extrinsics'][index],
                                             data['intrinsics'][index], size)
            foreground = data['masks'][index]
            scores.append(float((mask & foreground).sum() / (mask | foreground).sum()))
            image = Image.fromarray(rendered)
            path = out / f'view-{index}-variant-{vindex}.png'
            image.save(path)
            images[path.name] = module.sha(path)
            cells.append(image)
        for col, image in enumerate(cells):
            tile = ImageOps.contain(image, (276, 276))
            sheet.paste(tile, (col * 280 + 2, 60 + index * 300))
        draw.text((5, 42 + index * 300), f'Photo {index + 1}', fill='white')
        rows.append({'photoIndex': index, 'photoMaskIoU': scores})
    sheet.save(out / 'comparison.jpg', quality=95)
    module.save(out / 'report.json', {
        'status': 'rendered-visual-review-pending',
        'atUTC': datetime.now(timezone.utc).isoformat(),
        'scope': 'CPU diagnostic, own cameras and model resolution; not a controlled quantization quality gate',
        'scriptSHA256': module.sha(__file__), 'helperSHA256': module.sha(helper),
        'variants': variants, 'orderedSourceSHA256': sources, 'views': rows,
        'images': images, 'sheetSHA256': module.sha(out / 'comparison.jpg')})


if __name__ == '__main__':
    main()
