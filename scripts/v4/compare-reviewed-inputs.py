#!/usr/bin/env python3
"""Transfer an existing native review only when fresh input-case content is identical."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def members(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        return {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_run', type=Path)
    parser.add_argument('new_run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reviews', type=Path, nargs='+', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    old_bytes, new_bytes = args.old_run.read_bytes(), args.new_run.read_bytes()
    (output / 'old-run.json').write_bytes(old_bytes)
    (output / 'new-run.json').write_bytes(new_bytes)
    old_records = {r['name']: r for r in json.loads(old_bytes)['records']}
    new_records = {r['name']: r for r in json.loads(new_bytes)['records']}
    reviews = {}
    for path in args.reviews:
        for record in load(path)['records']:
            assert record['name'] not in reviews
            reviews[record['name']] = (path, record)
    report = {'status': 'running', 'requestedCases': args.names, 'records': [],
              'scriptSHA256': sha(__file__), 'oldRunSHA256': hashlib.sha256(old_bytes).hexdigest(),
              'newRunSHA256': hashlib.sha256(new_bytes).hexdigest(),
              'scope': 'Explicit completed cases only; exact content binding to earlier native review, no new visual observations',
              'limits': ['No whole-matrix pass implied', 'All earlier visual limitations remain',
                         'No UI/offline/performance or ground-truth geometry claim']}
    try:
        for name in args.names:
            old, new = old_records[name], new_records[name]
            assert old['status'] == new['status'] == 'core-input-and-export-passed'
            assert old['sourceIndices'] == new['sourceIndices']
            assert new['freshStageReuseCount'] == 0
            a, b = Path(old['job']), Path(new['job'])
            review_path, review = reviews[name]
            assert sha(a / 'bundle/model.usdz') == review['modelSHA256']
            for path, digest in review['images'].items():
                assert sha(review_path.parent / path) == digest
            old_inputs = load(a / 'inputs/dataset.json')['records']
            new_inputs = load(b / 'inputs/dataset.json')['records']
            assert len(old_inputs) == len(new_inputs) == old['photoCount'] == new['photoCount']
            mapping, photo_proof = {}, []
            for x, y in zip(old_inputs, new_inputs):
                assert x['name'] == y['name'] and x['sourceSHA256'] == y['sourceSHA256']
                assert sha(x['sourcePath']) == sha(y['sourcePath']) == y['sourceSHA256']
                mapping[x['id']] = y['id']
                images = {}
                for key in ('image', 'mask', 'fullImage', 'fullMask'):
                    images[key] = sha(a / 'inputs' / x[key])
                    assert images[key] == sha(b / 'inputs' / y[key])
                photo_proof.append({'sourceSHA256': y['sourceSHA256'], 'oldPhotoID': x['id'],
                                    'newPhotoID': y['id'], 'preparedFileSHA256': images})

            def mapped(value):
                if isinstance(value, str):
                    for x, y in mapping.items():
                        value = value.replace(x, y)
                    return value
                if isinstance(value, list):
                    return [mapped(v) for v in value]
                if isinstance(value, dict):
                    return {k: mapped(v) for k, v in value.items()}
                return value

            for x, y in zip(old_inputs, new_inputs):
                assert mapped({k: v for k, v in x.items() if k != 'sourcePath'}) == {
                    k: v for k, v in y.items() if k != 'sourcePath'}
            for file in ('cameras.json', 'texture-provenance.json', 'dataset.json'):
                assert mapped(load(a / 'bundle' / file)) == load(b / 'bundle' / file), file
            x, y = load(a / 'bundle/fusion.json'), load(b / 'bundle/fusion.json')
            for key in ('inputRun', 'elapsedSeconds'):
                x.pop(key)
                y.pop(key)
            assert x == y
            files = ['geometry.npz', 'model.glb', 'basecolor.png', 'uv.npz', 'atlas-surface.npz',
                     'geometry-face-map.npz', 'face-selection.npz', 'texture-source.png',
                     'texture-source-view.png', 'texture-quality.png']
            hashes = {file: sha(a / 'bundle' / file) for file in files}
            assert all(sha(b / 'bundle' / file) == digest for file, digest in hashes.items())
            usdz = members(a / 'bundle/model.usdz')
            assert usdz == members(b / 'bundle/model.usdz')
            report['records'].append({'name': name, 'photoCount': new['photoCount'], 'newJob': str(b),
                                     'newModelSHA256': sha(b / 'bundle/model.usdz'),
                                     'earlierVisualReviewSHA256': sha(review_path), 'earlierFindings': review['findings'],
                                     'photoIdentityProof': photo_proof, 'identicalFileSHA256': hashes,
                                     'identicalUSDZMembers': usdz, 'mappedCameraAndProvenanceEqual': True})
        report['status'] = 'fresh-content-identical-to-reviewed-inputs'
    except BaseException as error:
        report.update(status='review-transfer-not-proven', error=repr(error))
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': args.names}))


if __name__ == '__main__':
    main()
