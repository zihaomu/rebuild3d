#!/usr/bin/env python3
"""Verify a conventional project export after moving the lone USDZ file."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdGeom


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('export', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scriptSHA256': sha(Path(__file__)),
              'scope': 'Persisted conventional model, photo identity and isolated embedded assets; no UI/visual claim'}
    try:
        manifest = json.loads((args.project / 'project.json').read_text())
        model = manifest['model']
        assert not model.get('approximation')
        assert model['photoIDs'] == [p['id'] for p in manifest['photos']]
        for photo in manifest['photos']:
            assert sha(args.project / photo['imagePath']) == photo['metadata']['contentSHA256']
        snapshot = json.loads((args.project / model['inputSnapshotPath']).read_text())
        assert snapshot['runID'] == model['runID']
        assert [p['photo']['id'] for p in snapshot['inputs']] == model['photoIDs']
        digest = sha(args.project / model['path'])
        assert sha(args.export) == digest
        isolated = args.output / 'isolated'
        isolated.mkdir()
        relocated = isolated / args.export.name
        shutil.copy2(args.export, relocated)
        assert sha(relocated) == digest and list(isolated.iterdir()) == [relocated]
        with zipfile.ZipFile(relocated) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in names)
            images = []
            for name in names:
                if Path(name).suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    with Image.open(io.BytesIO(archive.read(name))) as image:
                        image.load()
                        images.append({'member': name, 'size': list(image.size), 'mode': image.mode,
                                       'decodedRGBSHA256': hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()})
        assert images
        stage = Usd.Stage.Open(str(relocated.resolve()))
        assert stage
        assets, meshes = [], []
        for prim in stage.Traverse():
            for attribute in prim.GetAttributes():
                value = attribute.Get()
                if isinstance(value, Sdf.AssetPath):
                    assert value.resolvedPath.startswith(str(relocated.resolve()) + '['), value
                    member = value.resolvedPath.split('[', 1)[1].rstrip(']')
                    assert member in names
                    assets.append({'attribute': str(attribute.GetPath()), 'authored': value.path, 'member': member})
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                vertices = np.asarray(mesh.GetPointsAttr().Get())
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                assert vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
                assert indices.size and counts.size and counts.sum() == indices.size
                assert indices.min() >= 0 and indices.max() < len(vertices) and counts.min() >= 3
                meshes.append({'path': str(prim.GetPath()), 'vertices': len(vertices), 'faces': len(counts),
                               'bounds': [vertices.min(0).tolist(), vertices.max(0).tolist()]})
        assert assets and meshes
        report.update(status='passed-artifact-validation', modelSHA256=digest, photoCount=len(manifest['photos']),
                      photoIDs=model['photoIDs'], inputSnapshotSHA256=sha(args.project / model['inputSnapshotPath']),
                      project=str(args.project.resolve()), export=str(args.export.resolve()),
                      relocatedUSDZ=str(relocated.resolve()), members=names, images=images,
                      embeddedAssets=assets, meshes=meshes)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'photoCount': report['photoCount'],
                      'embeddedImages': len(report['images']), 'meshCount': len(report['meshes'])}))


if __name__ == '__main__':
    main()
