#!/usr/bin/env python3
"""Create an immutable, standalone v4 runtime from verified local components.

The base runtime is read-only. Copying a checkpoint never converts it at install
or generation time. Output remains a local candidate, not a release upload.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Runtime'))
from rebuild3d_worker.infer import verify_repository
from rebuild3d_worker.model_weights import verify_weights
from rebuild3d_worker.mlx_block import FP16_LINEAR_POLICY, FP32_LINEAR_POLICY


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def safe_path(root,relative):
    path=root/relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Unsafe runtime component path: '+relative)
    return path


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--base-runtime',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--xatlas-extension',type=Path,help='Optional reviewed split-phase native extension built with xatlas-profile.patch')
    p.add_argument('--precompile-bytecode',action='store_true',
                   help='Precompile retained Python sources with checked hashes; include caches in component integrity checks')
    a=p.parse_args();out=a.output.resolve();base=a.base_runtime.resolve()
    if out.exists():raise ValueError('Runtime output must be new')
    info=verify_weights(a.weights)
    if info.get('format')!='rebuild3d-vggt-mlx' or info.get('headPrecision')!='fp16-parameters-fp32-position-and-activation-v1':
        raise ValueError('Expected the validated W8 + sensitive-FP32/head-FP16 candidate')
    groups = {(m['bits'], m['groupSize']) for m in info['quantizedModules'].values()}
    linear_policy = info.get('linearComputePolicy', FP16_LINEAR_POLICY)
    if not ((groups == {(8, 64)} and linear_policy == FP16_LINEAR_POLICY)
            or (groups == {(8, 32)} and linear_policy == FP32_LINEAR_POLICY)):
        raise ValueError('Unsupported production quantization policy')
    versions={name:importlib.metadata.version(name) for name in ('mlx','mlx-metal','torch')}
    if versions['mlx']!=info['runtimeMLXVersion'] or versions['mlx-metal']!=info['runtimeMLXVersion'] or versions['torch']!=info['runtimeTorchVersion']:
        raise ValueError('Local dependencies do not match the checkpoint runtime ABI')
    manifest=json.loads((base/'runtime.json').read_text())
    if manifest['formatVersion']!=1 or manifest['platform']!='macos-arm64':raise ValueError('Unsupported base runtime')
    entries=manifest['artifacts']
    if len({e['path'] for e in entries})!=len(entries):raise ValueError('Duplicate base component paths')
    for entry in entries:
        path=safe_path(base,entry['path'])
        if not path.is_file() or path.stat().st_size!=entry['byteCount'] or sha(path)!=entry['sha256']:
            raise ValueError('Frozen base component changed: '+entry['path'])
    verify_repository(base/'vggt')
    out.mkdir(parents=True)
    for entry in entries:
        name=entry['path']
        if name.startswith(('models/','worker/')):continue
        # Rebuild caches from this output's sources with its own interpreter.
        if name=='bytecode-manifest.json' or name.endswith('.pyc') or '__pycache__' in Path(name).parts:continue
        copy(base/name,out/name)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store')
    shutil.copytree(ROOT/'Runtime/rebuild3d_worker',out/'worker/rebuild3d_worker',ignore=ignore)
    copy(base/'worker/rebuild3d_worker/raster.dylib',out/'worker/rebuild3d_worker/raster.dylib')
    uv_library=out/'worker/rebuild3d_worker/uv_layout.dylib'
    uv_command=['xcrun','clang++','-O2','-std=c++17','-dynamiclib','-Wl,-install_name,@rpath/uv_layout.dylib',
                str(ROOT/'Runtime/rebuild3d_worker/uv_layout.cpp'),'-o',str(uv_library)]
    subprocess.run(uv_command,check=True)
    uv_library_build={'command':uv_command,'sha256':sha(uv_library),
        'sourceSHA256':{name:sha(ROOT/'Runtime/rebuild3d_worker'/name) for name in ('uv_layout.cpp','uv_layout.h')},
        'compiler':subprocess.check_output(['xcrun','clang++','--version'],text=True)}
    copy(a.weights,out/'models/vggt-1b.safetensors')
    copy(a.weights.with_suffix('.json'),out/'models/vggt-1b.json')
    site=out/'python/lib/python3.12/site-packages'
    uv_extension=None
    if a.xatlas_extension:
        if a.xatlas_extension.name!='xatlas.cpython-312-darwin.so':raise ValueError('Unexpected xatlas extension ABI')
        copy(a.xatlas_extension,site/a.xatlas_extension.name)
        uv_extension={'sha256':sha(a.xatlas_extension),'patchSHA256':sha(ROOT/'scripts/v4/xatlas-profile.patch'),
                      'bindingRevision':'24d7e3e99e498eb11a94f5efd54fd9f754f2b88a',
                      'xatlasRevision':'f700c7790aaa030e794b52ba7791a05c085faf0c'}
        copy(ROOT/'scripts/v4/xatlas-profile.patch',out/'xatlas-profile.patch')
        copy(ROOT/'scripts/v4/xatlas-NOTICES.txt',out/'xatlas-NOTICES.txt')
        copy(ROOT/'scripts/v4/xatlas-source-lock.json',out/'xatlas-source-lock.json')
    copied=set(); omitted_cli=[]
    for name in ('mlx','mlx-metal'):
        dist=importlib.metadata.distribution(name)
        console_names={entry.name for entry in dist.entry_points if entry.group=='console_scripts'}
        for relative in dist.files:
            if '__pycache__' in relative.parts or str(relative).endswith('.pyc'):continue
            if relative.parts[:4]==('..','..','..','bin') and len(relative.parts)==5 and relative.name in console_names:
                # Distributed launchers are not part of this single-process app;
                # their installed shebangs point at the development venv.
                omitted_cli.append({'distribution':name,'path':str(relative),'reason':'unused distributed console entry point'})
                continue
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unexpected wheel path: '+str(relative))
            source=dist.locate_file(relative)
            if source.is_symlink() or not source.is_file():raise ValueError('Missing/non-regular wheel member')
            target=site/relative
            if str(relative) in copied and sha(target)!=sha(source):raise ValueError('Dependency wheel file conflict')
            copy(source,target);copied.add(str(relative))
    copy(ROOT/'scripts/v4/requirements-mlx.lock',out/'requirements-mlx.lock')
    notice=out/'NOTICE.md'
    notice.write_text(notice.read_text()+
        '\n## v4 local quantized candidate\n\n'
        'MLX and MLX Metal are pinned to 0.32.2; their MIT license notices are retained in python/lib/python3.12/site-packages/. '
        'The VGGT backbone algebra follows the pinned VGGT layer sources; original code and license remain in vggt/. '
        'Derived weight lineage, packing, precision and SHA-256 are recorded in models/vggt-1b.json. '
        'Quantization does not change the original weight CC-BY-NC-4.0 restriction. '
        'This package contains one prepacked production candidate and no input photographs.\n')
    source={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'Runtime/rebuild3d_worker').glob('*')) if f.is_file()}
    bytecode=None
    if a.precompile_bytecode:
        compiler=ROOT/'scripts/v4/precompile-runtime.py'
        subprocess.run([str(out/'python/bin/python3.12'),'-I','-B',str(compiler),str(out)],check=True)
        compiled=json.loads((out/'bytecode-manifest.json').read_text())
        bytecode={key:compiled[key] for key in ('policy','compiledFiles','byteCount','compilerSHA256')}
        bytecode['manifestSHA256']=sha(out/'bytecode-manifest.json')
    save(out/'v4-build.json',{'formatVersion':1,'baseRuntimeManifestSHA256':sha(base/'runtime.json'),
         'packagerSHA256':sha(__file__),'workerSource':source,'runtimeVersions':versions,
         'backendID':info['backendID'],'derivedWeightSHA256':info['derivedWeightSHA256'],
         'weightManifestSHA256':sha(a.weights.with_suffix('.json')),
         'omittedConsoleEntrypoints':omitted_cli,
         'xatlasExtension':uv_extension,
         'uvLayoutLibrary':uv_library_build,
         'bytecode':bytecode,
         'candidateStatus':'full application acceptance pending'})
    files=[{'path':str(f.relative_to(out)),'byteCount':f.stat().st_size,'sha256':sha(f)}
           for f in sorted(out.rglob('*')) if f.is_file()]
    if any(f.is_symlink() for f in out.rglob('*')):raise ValueError('Standalone runtime must not contain symlinks')
    result={'formatVersion':1,'platform':'macos-arm64','pythonVersion':manifest.get('pythonVersion'),
            'vggtRevision':manifest.get('vggtRevision'),'weightSHA256':info['derivedWeightSHA256'],
            'backendID':info['backendID'],'runtimeVersions':versions,'artifacts':files}
    save(out/'runtime.json',result)
    # A relocated interpreter must resolve every dependency within this runtime.
    code="import importlib.util,json,sys;print(json.dumps({'prefix':sys.prefix,'modules':{n:importlib.util.find_spec(n).origin for n in ['torch','mlx.core','numpy','pxr','xatlas']}}))"
    probe=json.loads(subprocess.check_output([str(out/'python/bin/python3.12'),'-I','-B','-c',code],text=True))
    if any(not Path(path).resolve().is_relative_to(out) for path in [probe['prefix'],*probe['modules'].values()]):
        raise ValueError('Runtime still depends on an external Python installation')
    total=sum(f['byteCount'] for f in files)+(out/'runtime.json').stat().st_size
    print(json.dumps({'output':str(out),'bytes':total,'artifacts':len(files),'relocationProbe':probe,
                      'appSizeGate':'pending; include app executable and resources'},indent=2))


if __name__=='__main__':main()
