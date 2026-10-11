#!/usr/bin/env python3
"""Assemble a local v4 app from an explicit Swift binary and verified runtime.

Build the Swift executable separately with `swift build -c release --product
Rebuild3D`, then pass that binary. This script never claims to compile Swift.
"""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runtime',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--bundle-id', default='org.rebuild3d.v4candidate',
                   help='Use a distinct local identity when testing several candidates')
    p.add_argument('--display-name', default='Rebuild3D v4 Candidate')
    a=p.parse_args();runtime=a.runtime.resolve();out=a.output.resolve()
    if not re.fullmatch(r'[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+', a.bundle_id):
        raise ValueError('Expected a reverse-DNS bundle identifier')
    if not a.display_name.strip():raise ValueError('Expected a nonempty display name')
    if out.exists() or out.suffix!='.app':raise ValueError('Expected a new .app path')
    manifest=json.loads((runtime/'runtime.json').read_text())
    if manifest.get('backendID')!='mlx-aggregator-pytorch-heads-v1':raise ValueError('Expected the v4 MLX runtime')
    for entry in manifest['artifacts']:
        file=runtime/entry['path']
        if not file.resolve().is_relative_to(runtime) or not file.is_file() or file.stat().st_size!=entry['byteCount'] or sha(file)!=entry['sha256']:
            raise ValueError('Runtime artifact failed verification: '+entry['path'])
    arches=subprocess.check_output(['lipo','-archs',str(a.binary)],text=True).strip()
    if arches!='arm64':raise ValueError('Expected Apple Silicon executable')
    source_sha=sha(a.binary)
    macos=out/'Contents/MacOS';resources=out/'Contents/Resources'
    macos.mkdir(parents=True);resources.mkdir()
    shutil.copy2(a.binary,macos/'Rebuild3D')
    info=plistlib.loads((ROOT/'Config/Info.plist').read_bytes())
    # Distinct identifiers also distinguish side-by-side local candidates.
    info.update(CFBundleIdentifier=a.bundle_id,CFBundleDisplayName=a.display_name,CFBundleName=a.display_name)
    (out/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    shutil.copy2(ROOT/'LICENSE',resources/'LICENSE')
    shutil.copy2(ROOT/'Vendor/Photogrammetry/LICENSE',resources/'Photogrammetry-MIT-LICENSE')
    subprocess.run(['cp','-cR',str(runtime),str(resources/'GenerationRuntime')],check=True)
    build={'kind':'local candidate; not a published release','sourceBinary':str(a.binary.resolve()),
           'bundleIdentifier':a.bundle_id,'displayName':a.display_name,
           'sourceBinarySHA256':source_sha,'swiftCompilation':'external/prebuilt; not performed by this packager',
           'runtimeManifestSHA256':sha(runtime/'runtime.json'),'packagerSHA256':sha(__file__)}
    (resources/'candidate-build.json').write_text(json.dumps(build,indent=2)+'\n')
    subprocess.run(['codesign','--force','--sign','-',str(out)],check=True)
    subprocess.run(['codesign','--verify','--strict',str(out)],check=True)
    total=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    result={**build,'output':str(out),'expandedAppBytes':total,'under3500000000Bytes':total<=3500000000,
            'remaining':'one-click generation, relocation/offline, recovery, quality and final performance acceptance'}
    out.with_suffix('.build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if total>3500000000:raise ValueError('Expanded app exceeds P0 size target')


if __name__=='__main__':main()
