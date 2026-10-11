#!/usr/bin/env python3
"""Compare aligned sparse predictions; numeric gates never replace visual review."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def angle(matrix):
    return np.degrees(np.arccos(np.clip((np.trace(matrix, axis1=-2, axis2=-1)-1)/2, -1, 1)))


def rotations(extrinsics):
    r=extrinsics[:,:,:3].astype('float64')
    if np.max(np.abs(r@r.transpose(0,2,1)-np.eye(3)))>1e-3 or (np.linalg.det(r)<=0).any():
        raise ValueError('Invalid camera rotation matrix')
    # Remove float32 orthogonality roundoff before acos(trace): identical
    # rotations otherwise produce an artificial ~0.03 degree error floor.
    u,_,vt=np.linalg.svd(r)
    return u@vt


def verify_relocated_inputs(reference, candidate, old_config, new_config):
    """Only sourcePath may vary; bind both full manifests and working bytes."""
    documents, artifact_sets = [], []
    for inference, config in ((reference, old_config), (candidate, new_config)):
        root = inference.parent / 'inputs'
        raw = (root / 'dataset.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != config['datasetSHA256']:
            raise ValueError('Relocated dataset does not match recorded SHA256')
        document = json.loads(raw)
        records = document['records']
        if [r['id'] for r in records] != config['photoIDs'] or [r['sourceSHA256'] for r in records] != config['sourceSHA256']:
            raise ValueError('Relocated dataset identity/order differs')
        artifacts = []
        for record in records:
            for key in ('image', 'mask'):
                path = (root / record[key]).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError('Missing or foreign relocated working artifact')
                artifacts.append({'photoID': record['id'], 'kind': key,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        artifact_sets.append(artifacts)
        documents.append(document)
    paths = [{'photoID': a['id'], 'reference': a['sourcePath'], 'candidate': b['sourcePath']}
             for a, b in zip(documents[0]['records'], documents[1]['records'])]
    normalized = []
    for document in documents:
        normalized.append({**document, 'records': [{k: v for k, v in r.items() if k != 'sourcePath'}
                                                    for r in document['records']]})
    if normalized[0] != normalized[1] or artifact_sets[0] != artifact_sets[1]:
        raise ValueError('Relocated inputs changed beyond sourcePath')
    return {'mode': 'verified-relocation', 'referenceDatasetSHA256': old_config['datasetSHA256'],
            'candidateDatasetSHA256': new_config['datasetSHA256'], 'sourcePaths': paths,
            'identicalWorkingArtifacts': artifact_sets[0]}


def compare(reference, candidate, thresholds, *, relocated_inputs=False):
    old_config = json.loads((reference / "config.json").read_text())
    new_config = json.loads((candidate / "config.json").read_text())
    for key in ("photoIDs", "sourceSHA256"):
        if old_config[key] != new_config[key]:
            raise ValueError("Input identity/order differs: " + key)
    input_proof = {'mode': 'identical-dataset-SHA256', 'datasetSHA256': old_config['datasetSHA256']}
    if relocated_inputs:
        input_proof = verify_relocated_inputs(reference, candidate, old_config, new_config)
    elif old_config['datasetSHA256'] != new_config['datasetSHA256']:
        raise ValueError('Input identity/order differs: datasetSHA256')
    if json.loads((reference / "pixel-transforms.json").read_text()) != json.loads((candidate / "pixel-transforms.json").read_text()):
        raise ValueError("Pixel transforms differ")
    with np.load(reference / "predictions.npz") as f:
        a = {k: f[k] for k in f.files}
    with np.load(candidate / "predictions.npz") as f:
        b = {k: f[k] for k in f.files}
    if set(a) != set(b) or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("Prediction schema or shape mismatch")
    if any(not np.isfinite(b[k]).all() for k in b):
        raise ValueError("Nonfinite predictions")
    for key in ("rgb", "masks"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError("Preprocessing differs: " + key)
    mask = a["masks"].astype(bool)
    valid_a = np.isfinite(a["depth"]) & (a["depth"] > 0) & mask
    valid_b = np.isfinite(b["depth"]) & (b["depth"] > 0) & mask
    both = valid_a & valid_b
    if not both.any() or not mask.any():
        raise ValueError("No valid common foreground depth")
    # A single scene scale for all cameras/depth maps, never one scale per view.
    scale = float(np.median(a["depth"][both].astype('float64') / b["depth"][both]))
    depth = np.abs(b["depth"].astype('float64')*scale-a["depth"]) / np.maximum(a["depth"], 1e-12)
    ra, rb = rotations(a['extrinsics']), rotations(b['extrinsics'])
    # Choose the one global orientation closest to all corresponding cameras.
    u, _, vt = np.linalg.svd(np.einsum('nji,njk->nik', ra, rb).mean(axis=0))
    fix = np.diag([1,1,np.linalg.det(u@vt)])
    rotation = u @ fix @ vt
    ca = -np.einsum('nji,nj->ni', ra, a['extrinsics'][:, :, 3])
    cb = -np.einsum('nji,nj->ni', rb, b['extrinsics'][:, :, 3])
    translation = ca.mean(0) - scale * (rotation @ cb.mean(0))
    centers = scale * (cb @ rotation.T) + translation
    reprojection, behind, sampled = [], 0, 0
    for i in range(len(ra)):
        # Deterministic foreground grid, evaluated with the same scene transform.
        y, x = np.nonzero(valid_a[i][::8, ::8]); y, x = y*8, x*8
        pixels = np.stack((x,y,np.ones_like(x)),axis=-1).astype('float64')
        camera_points = (pixels @ np.linalg.inv(a['intrinsics'][i]).T) * a['depth'][i,y,x,None]
        world_points = (camera_points-a['extrinsics'][i,:,3]) @ ra[i]
        candidate_world = (world_points-translation) @ rotation / scale
        candidate_camera = candidate_world @ rb[i].T + b['extrinsics'][i,:,3]
        visible = candidate_camera[:,2] > 1e-8
        projected = candidate_camera[visible] @ b['intrinsics'][i].T
        reprojection.extend(np.linalg.norm(projected[:,:2]/projected[:,2,None]-pixels[visible,:2],axis=1))
        sampled += len(pixels); behind += int((~visible).sum())
    relative = []
    for i in range(len(ra)):
        for j in range(i+1, len(ra)):
            relative.append(angle((rb[j] @ rb[i].T) @ (ra[j] @ ra[i].T).T))
    fa, fb = a['intrinsics'][:, [0,1], [0,1]], b['intrinsics'][:, [0,1], [0,1]]
    if (fa <= 0).any() or (fb <= 0).any():
        raise ValueError("Invalid focal length")
    focal = np.abs(fb-fa)/fa
    valid_drop = float(valid_a.sum()-valid_b.sum())/int(mask.sum())
    metrics = {"relativeRotationDegrees": stats(relative), "focalRelative": stats(focal),
               "depthRelative": stats(depth[both]), "validDepthFractionDrop": valid_drop,
               "cameraCenterDistance": stats(np.linalg.norm(centers-ca, axis=1)),
               "referenceCameraExtent": float(np.linalg.norm(ca-ca.mean(0), axis=1).max()),
               "projectionDisplacementModelPixels": stats(reprojection) if reprojection else None,
               "projectionBehindFraction": behind/max(sampled,1),
               "confidenceRelative": stats((np.abs(b['confidence']-a['confidence'])/np.maximum(np.abs(a['confidence']),1e-12))[both]),
               "perPhotoDepth": [{"photoID": p, **stats(depth[i][both[i]])} for i,p in enumerate(old_config['photoIDs'])]}
    gates = {"rotation": metrics['relativeRotationDegrees']['p95'] <= thresholds['rotationP95Degrees'],
             "focal": metrics['focalRelative']['p95'] <= thresholds['focalRelativeP95'],
             "depthMedian": metrics['depthRelative']['median'] <= thresholds['depthRelativeMedian'],
             "depthP95": metrics['depthRelative']['p95'] <= thresholds['depthRelativeP95'],
             "validDepth": valid_drop <= thresholds['validDepthFractionDrop']}
    return {"reference": str(reference), "candidate": str(candidate), "thresholds": thresholds,
            "inputIdentityProof": input_proof,
            "scope": "Inference numeric gates only; mesh, UV, texture, fixed views and UI review still required",
            "sceneSimilarity": {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
                                "method": "one foreground depth ratio median + camera orientation mean + center translation"},
            "metrics": metrics, "gates": gates, "numericGatesPassed": all(gates.values()),
            "bitwiseEqualArrays": {key: bool(np.array_equal(a[key], b[key])) for key in a}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path); p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--relocated-inputs', action='store_true',
                   help='Verify sibling inputs folders, permitting only original sourcePath changes')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Comparison output must be new')
    result = compare(args.reference, args.candidate, json.loads(args.protocol.read_text())['thresholds'],
                     relocated_inputs=args.relocated_inputs)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('gates', 'numericGatesPassed', 'metrics')}))
    return 0 if result['numericGatesPassed'] else 1


if __name__ == '__main__':
    sys.exit(main())
