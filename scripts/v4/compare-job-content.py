#!/usr/bin/env python3
"""Compare complete saved jobs after an execution-only change, without rendering.

Photo UUIDs and their absolute source paths may change when importing/saving.
Only those identities are normalized. Geometry, textures and prediction arrays
must be identical. Earlier visual findings remain applicable only to compared
content; this is not a new visual, UI, reliability or performance acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def members(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()}


def compare(old, new, old_project, new_project):
    before, after = (read(p / 'inputs/dataset.json')['records'] for p in (old, new))
    assert len(before) == len(after) and len(before) > 0
    identities, photos = {}, []
    manifests = [read(p / 'project.json') for p in (old_project, new_project)]
    records = [{p['id']: p for p in m['photos']} for m in manifests]
    for a, b in zip(before, after):
        assert (a['name'], a['sourceSHA256']) == (b['name'], b['sourceSHA256'])
        identities[a['id']] = b['id']
        for root, index, item in zip((old_project, new_project), records, (a, b)):
            assert sha(root / index[item['id']]['imagePath']) == item['sourceSHA256']
        prepared = {}
        for key in ('image', 'mask', 'fullImage', 'fullMask'):
            prepared[key] = sha(old / 'inputs' / a[key])
            assert prepared[key] == sha(new / 'inputs' / b[key]), key
        photos.append({'name': b['name'], 'sourceSHA256': b['sourceSHA256'],
                       'oldID': a['id'], 'newID': b['id'], 'preparedSHA256': prepared})

    def mapped(value):
        if isinstance(value, str):
            for a, b in identities.items():
                value = value.replace(a, b)
            return value
        if isinstance(value, list):
            return [mapped(v) for v in value]
        if isinstance(value, dict):
            return {k: mapped(v) for k, v in value.items()}
        return value

    for a, b in zip(before, after):
        assert mapped({k: v for k, v in a.items() if k != 'sourcePath'}) == {
            k: v for k, v in b.items() if k != 'sourcePath'}
    json_files = ['inference/cameras.json', 'inference/pixel-transforms.json',
                  'bundle/cameras.json', 'bundle/texture-provenance.json', 'bundle/dataset.json']
    for name in json_files:
        assert mapped(read(old / name)) == read(new / name), name
    arrays = {}
    with np.load(old / 'inference/predictions.npz', allow_pickle=False) as a, \
            np.load(new / 'inference/predictions.npz', allow_pickle=False) as b:
        assert a.files == b.files
        for name in a.files:
            assert a[name].dtype == b[name].dtype and np.array_equal(a[name], b[name]), name
            arrays[name] = {'shape': list(a[name].shape), 'dtype': str(a[name].dtype)}
    files = sorted(p.name for p in (old / 'bundle').iterdir()
                   if p.suffix in ('.npz', '.png', '.glb', '.usdc'))
    assert files == sorted(p.name for p in (new / 'bundle').iterdir()
                           if p.suffix in ('.npz', '.png', '.glb', '.usdc'))
    hashes = {}
    for name in files:
        hashes[name] = sha(old / 'bundle' / name)
        assert hashes[name] == sha(new / 'bundle' / name), name
    archives = {}
    for name in ('model.usdz', 'provenance.usdz', 'texture-sources.usdz'):
        archives[name] = members(old / 'bundle' / name)
        assert archives[name] == members(new / 'bundle' / name), name
    events = [json.loads(s) for s in (new / 'events.jsonl').read_text().splitlines()]
    assert read(new / 'state.json')['status'] == 'completed'
    return {'photos': photos, 'predictionArrays': arrays, 'mappedJSONEqual': json_files,
            'identicalFiles': hashes, 'identicalArchiveMembers': archives,
            'newModelSHA256': sha(new / 'bundle/model.usdz'),
            'newStageReuseCount': sum(e['event'] == 'stage-reused' for e in events),
            'backendConfig': read(new / 'inference/config.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old_job', 'new_job', 'old_project', 'new_project', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scriptSHA256': sha(__file__),
              'createdAtUTC': datetime.now(timezone.utc).isoformat(),
              'oldJob': str(args.old_job.resolve()), 'newJob': str(args.new_job.resolve()),
              'limits': ['Exact compared content only; previous visual limitations remain.',
                         'No new visual/UI/reliability/performance pass implied.']}
    try:
        report.update(compare(args.old_job, args.new_job, args.old_project, args.new_project))
        report['status'] = 'complete-job-content-identical'
    except BaseException as error:
        report.update(status='content-equality-not-proven', error=repr(error))
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'files': len(report['identicalFiles']),
                      'predictionArrays': len(report['predictionArrays'])}))


if __name__ == '__main__':
    main()
