#!/usr/bin/env python3
"""Observe packaged fusion normal repairs on an existing inference, without changing results.

Run with the candidate Python and PYTHONPATH pointing at its worker directory.
This is a diagnostic replay, not a fresh reconstruction or performance sample.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import trimesh
from rebuild3d_worker import fuse


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def describe(mesh):
    return {'faces': len(mesh.faces), 'vertices': len(mesh.vertices),
            'watertight': bool(mesh.is_watertight), 'windingConsistent': bool(mesh.is_winding_consistent),
            'trimeshVolume': float(mesh.volume), 'bounds': mesh.bounds.tolist(),
            'facesSHA256': array_sha(mesh.faces), 'verticesSHA256': array_sha(mesh.vertices)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    output, job = args.output.resolve(), args.job.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'job': str(job), 'scriptSHA256': sha(Path(__file__)),
              'packagedFuseSHA256': sha(Path(fuse.__file__)), 'trimeshVersion': trimesh.__version__,
              'inferenceSHA256': sha(job / 'inference/predictions.npz'), 'repairs': [],
              'scope': 'Diagnostic replay with observed normal-repair calls; same inference and fusion arguments'}
    original = trimesh.repair.fix_normals

    def observed(mesh, *positional, **keywords):
        if len(mesh.faces) < 10000:
            return original(mesh, *positional, **keywords)
        before = describe(mesh)
        result = original(mesh, *positional, **keywords)
        item = {'arguments': repr((positional, keywords)), 'before': before, 'after': describe(mesh)}
        report['repairs'].append(item)
        print(json.dumps({'normalRepair': item}), flush=True)
        return result

    trimesh.repair.fix_normals = observed
    start = time.monotonic()
    try:
        resolution = json.loads((job / 'fusion/report.json').read_text())['resolution']
        sys.argv = ['packaged-fuse-diagnostic', str(job / 'inference'), str(output / 'fusion'), '--resolution', str(resolution)]
        fuse.main()
        with np.load(job / 'fusion/face-provenance.npz') as a, np.load(output / 'fusion/face-provenance.npz') as b:
            identical = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
            assert set(a.files) == set(b.files)
        report['exactArrayEqualityToOriginal'] = identical
        assert all(identical.values()), 'Observational replay changed fusion arrays'
        report['status'] = 'diagnostic-replay-identical'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['elapsedSecondsDiagnosticOnly'] = time.monotonic() - start
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
