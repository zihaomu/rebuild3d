#!/usr/bin/env python3
"""Verify actual application exports and an isolated USDZ copy.

This checks persisted bytes and embedded assets. UI save/reopen/display and
offline execution require separate observations; this report never claims them.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import zipfile

from PIL import Image
from pxr import Sdf, Usd


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or foreign artifact: {relative}')
    return path


def verify_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', data)
    assert magic == 0x46546c67 and version == 2 and length == len(data)
    chunks, offset = {}, 12
    while offset < length:
        size, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        assert offset + size <= length and kind not in chunks
        chunks[kind] = data[offset:offset + size]
        offset += size
    assert offset == length
    document = json.loads(chunks[0x4e4f534a])
    binary = chunks[0x004e4942]
    assert len(document['buffers']) == 1
    assert 'uri' not in document['buffers'][0]
    assert document['buffers'][0]['byteLength'] <= len(binary)
    images = []
    for item in document['images']:
        assert 'uri' not in item and 'bufferView' in item
        view = document['bufferViews'][item['bufferView']]
        assert view['buffer'] == 0
        begin, size = view.get('byteOffset', 0), view['byteLength']
        assert 0 <= begin < begin + size <= len(binary)
        with Image.open(io.BytesIO(binary[begin:begin + size])) as image:
            image.load()
            images.append({'size': list(image.size), 'mode': image.mode,
                           'rgbPixelSHA256': hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()})
    assert images and document['meshes']
    for mesh in document['meshes']:
        for primitive in mesh['primitives']:
            assert all(x in primitive['attributes'] for x in ('POSITION', 'NORMAL', 'TEXCOORD_0'))
            material = document['materials'][primitive['material']]
            texture = material['pbrMetallicRoughness']['baseColorTexture']['index']
            assert 0 <= document['textures'][texture]['source'] < len(images)
    return {'embeddedImages': images, 'externalURIs': [], 'meshCount': len(document['meshes'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('export', type=Path)
    parser.add_argument('new_output', type=Path)
    args = parser.parse_args()
    args.new_output.mkdir(parents=True, exist_ok=False)
    project = json.loads((args.project / 'project.json').read_text())
    model = project['model']
    research = child(args.project, model['approximation']['bundlePath']).parent
    companion = args.export.with_suffix('.rebuild3d-result')
    bundle = json.loads((companion / 'bundle.json').read_text())
    assert (research / 'bundle.json').read_bytes() == (companion / 'bundle.json').read_bytes()
    artifacts = []
    for item in bundle['artifacts']:
        actual = child(companion, item['path'])
        source = child(research, item['path'])
        assert actual.stat().st_size == item['byteCount']
        digest = sha(actual)
        assert digest == item['sha256'] == sha(source)
        artifacts.append({'path': item['path'], 'sha256': digest, 'byteCount': item['byteCount']})
    photo_ids = [x['id'] for x in project['photos']]
    assert photo_ids == model['photoIDs']
    photos = []
    for photo in project['photos']:
        digest = sha(child(args.project, photo['imagePath']))
        assert digest == photo['metadata']['contentSHA256']
        photos.append({'id': photo['id'], 'name': photo['originalName'], 'sha256': digest})
    assert [(x['name'], x['sha256']) for x in photos] == [
        (x['name'], x['sha256']) for x in bundle['sourcePhotos']]
    cameras = json.loads((companion / 'cameras.json').read_text())['cameras']
    assert [x['id'] for x in cameras] == photo_ids
    assert [x['name'] for x in cameras] == [x['name'] for x in photos]
    snapshot = child(args.project, model['inputSnapshotPath'])
    model_sha = sha(child(args.project, model['path']))
    assert model_sha == sha(args.export) == sha(companion / 'model.usdz')
    isolated = args.new_output / 'isolated'
    isolated.mkdir()
    relocated = isolated / args.export.name
    shutil.copy2(args.export, relocated)
    assert sha(relocated) == model_sha
    assert list(isolated.iterdir()) == [relocated]
    with zipfile.ZipFile(relocated) as archive:
        assert archive.testzip() is None
        members = archive.namelist()
        assert all(not Path(x).is_absolute() and '..' not in Path(x).parts for x in members)
    stage = Usd.Stage.Open(str(relocated.resolve()))
    assert stage
    assets = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath):
                assert value.resolvedPath.startswith(str(relocated.resolve()) + '[')
                member = value.resolvedPath.split('[', 1)[1].rstrip(']')
                assert member in members
                assets.append({'attribute': str(attribute.GetPath()), 'authored': value.path,
                               'resolved': value.resolvedPath})
    assert assets
    glb = {name: verify_glb(companion / name) for name in
           ('model.glb', 'provenance.glb', 'texture-sources.glb')}
    # The atlas preserves its aspect ratio; the configured long edge is 4K.
    atlas = json.loads((companion / 'uv-report.json').read_text())
    expected_size = [atlas['width'], atlas['height']]
    assert max(expected_size) == 4096
    for name, texture in [('model.glb', 'basecolor.png'),
                          ('provenance.glb', 'geometry-source.png'),
                          ('texture-sources.glb', 'texture-source-display.png')]:
        embedded = glb[name]['embeddedImages'][0]
        assert embedded['size'] == expected_size
        with Image.open(companion / texture) as original:
            assert list(original.size) == expected_size
            assert embedded['rgbPixelSHA256'] == hashlib.sha256(original.convert('RGB').tobytes()).hexdigest()
    report = {'status': 'passed-artifact-validation', 'scriptSHA256': sha(Path(__file__)),
              'project': str(args.project.resolve()), 'export': str(args.export.resolve()),
              'companion': str(companion.resolve()), 'relocatedUSDZ': str(relocated.resolve()),
              'modelSHA256': model_sha, 'inputSnapshotSHA256': sha(snapshot),
              'photos': photos, 'artifacts': artifacts, 'cameraPhotoIDsMatch': True,
              'usdMembers': members, 'resolvedUSDAssets': assets, 'glb': glb,
              'limits': 'Artifact validation only; visual/UI and offline checks are separate.'}
    (args.new_output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'artifactCount': len(artifacts),
                      'photoCount': len(photos), 'relocatedUSDZ': report['relocatedUSDZ']}))


if __name__ == '__main__':
    main()
