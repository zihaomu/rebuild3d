"""Package the enhanced model with unchanged geometry provenance and texture evidence."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from common import sha,write_json

p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('candidate',type=Path)
p.add_argument('stage1',type=Path);p.add_argument('stage2',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);c=a.candidate;s=a.stage1/'reproduction-01'
dataset=json.loads((a.inputs/'dataset.json').read_text());fusion=json.loads((s/'fusion/report.json').read_text())
for r in dataset['records']:
    assert sha(r['sourcePath'])==r['sourceSHA256'] and sha(a.inputs/r['mask'])==r['maskSHA256']
files={name:c/name for name in ['model.usdz','model.glb','texture-sources.usdz','texture-sources.glb',
 'basecolor.png','texture-source.png','texture-quality.png','texture-source-display.png','texture-source-legend.png','texture-source-legend.json',
 'texture-provenance.json','geometry-face-map.npz','atlas-surface.npz','geometry.npz','uv.npz','face-selection.npz',
 'cameras.json','fusion.json','config.json','report.json','uv-audit.json','seam-report.json','asset-report.json','texture-sources-asset-report.json','view-preferences.json']}
files.update({'provenance.usdz':s/'result/provenance.usdz','provenance.glb':s/'result/provenance.glb',
 'geometry-provenance.npz':s/'fusion/face-provenance.npz','pixel-transforms.json':s/'inference/pixel-transforms.json',
 'dataset.json':a.inputs/'dataset.json','mask-corrections.json':a.inputs/'mask-corrections.json','mask-revision-report.json':a.inputs/'mask-revision-report.json',
 'baseline/baseline.json':a.stage2/'baseline/baseline.json','baseline/review-regions.json':a.stage2/'baseline/review-regions.json',
 'baseline/review-region-faces.npz':a.stage2/'baseline/review-region-faces.npz',
 'projection/report.json':a.stage2/'projection-audit-r2/report.json','camera-refinement/report.json':a.stage2/'camera-refinement-audit/report.json',
 'reproduction/comparison.json':a.stage2/'reproduction-01/comparison.json',
 'dependencies/requirements-macos.lock':a.stage2/'dependencies/requirements-macos.lock',
 'dependencies/xatlas-python-LICENSE.txt':a.stage2/'dependencies/xatlas-python-LICENSE.txt'})
for r in dataset['records']:files[r['mask']]=a.inputs/r['mask']
for directory in ['review','renders','source-renders']:
    for file in (c/directory).iterdir():
        if file.is_file():files[f'{directory}/{file.name}']=file
for file in Path('scripts/texture').iterdir():
    if file.is_file():files[f'reproduce/scripts/texture/{file.name}']=file
for name in ['prepare-texture-inputs.swift','render-texture-review.swift']:
    files[f'reproduce/scripts/{name}']=Path('scripts')/name
for relative,source in files.items():
    target=a.output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
readme='''# 七图佛像：第二阶段纹理增强来源包

普通模型：`model.usdz` / `model.glb`，照片颜色已内嵌。
几何来源：`provenance.usdz`，橙色为学习深度推测，紫色为轮廓补全；**全部几何仍是推测，比例不是实测尺寸**。
纹理来源：`texture-sources.usdz` / `texture-sources.glb`，灰色为没有可靠照片取色的外观填充，其他七色对应原照片；见 `texture-source-legend.png`。它与几何 Sources 是两种不同分类。

保留原 488824 面、原三维角点与 66480 个补全面，UV 接缝仅复制顶点。实际颜色纹理 4063×4096，来自 7 张完整正向 PNG（HEIC 明确 SDR/sRGB 8 位解码），没有生成式补脸、超分或新造雕纹。

`texture-provenance.json`、`geometry.npz`、`uv.npz`、`atlas-surface.npz` 和相机参数可从 atlas 像素重算原始面与源图位置；`texture-source.png` 的 0 表示外观填充或 UV 外部，结合 atlas 面索引区分。`texture-quality.png` 的 1 表示边缘/斜视风险，2 表示通过检查的内部照片取色；二者都不是测量真值。

照片支持约占有效 atlas 像素 72.98%，其余约 27.02% 使用低频外观填充；面积口径和误差边界见 `uv-audit.json`。原相机仍是学习估计，跨视角阴影差异和部分接缝仍在，细杆与遮挡部位几何仍粗糙。微调未通过独立保留点的精细混合门槛，因此保留主视图，不强行平均错位图案。

`review/` 含冻结的八处原图/旧版/新版对照、七视角和转台；输入照片也参与了生成，这些比较不是独立的三维真实性证明。`reproduction/comparison.json` 记录新目录复现：PNG、UV、颜色、来源图及 GLB 一致；USDZ 仅 ZIP 时间戳不同，内嵌场景和纹理字节一致。

在拥有相同 7 张原照片的项目副本中使用 Load Approximation 导入本目录。普通模式显示照片纹理，Sources 显示几何推测。纹理来源模型可单独打开。所有附件在 `bundle.json` 中逐文件校验，导出时应保留整个来源包。

本包不包含 HEIC 原件、VGGT 权重或可重新推理的模型权重。第一阶段使用的 VGGT-1B 研究权重许可仍为 CC-BY-NC-4.0；纹理增强未改变其许可边界。复现使用仓库记录的第一阶段几何和原始七图，具体命令见[第二阶段历史探索记录](https://github.com/zihaomu/rebuild3d/blob/a586616d5e1d014a1c038ab85550c2748930674d/doc/第二阶段-纹理增强探索记录-2026-10-08.md)。
'''
(a.output/'README.md').write_text(readme)
artifacts=[{'path':str(path.relative_to(a.output)),'sha256':sha(path),'byteCount':path.stat().st_size}
           for path in sorted(a.output.rglob('*')) if path.is_file()]
bundle={'formatVersion':1,'kind':'approximate','method':'VGGT inferred geometry + full-resolution photographic UV texture, stage 2',
 'createdAt':datetime.now(timezone.utc).isoformat(),'sourcePhotos':[{'name':r['name'],'sha256':r['sourceSHA256']} for r in dataset['records']],
 'model':'model.usdz','provenanceModel':'provenance.usdz','artifacts':artifacts,
 'triangleCount':fusion['triangles'],'completionTriangleCount':fusion['completionTriangles'],
 'limitations':['All geometry and cameras remain inferred; relative scale only.',
 'Orange/purple Sources retain learned-depth/silhouette geometry classes. Separate texture-sources model uses grey for appearance fill.',
 'About 73% of active atlas texels have photograph samples; the remainder is labelled low-frequency appearance fill.',
 'High-resolution color is not proof of accurate 3D placement, relightable albedo, or complete PBR recovery.',
 'Cross-view shadows, residual seams, rough thin parts and occlusions remain approximate.',
 'Original VGGT-1B research weights use CC-BY-NC-4.0; no weights bundled.']}
write_json(a.output/'bundle.json',bundle);print(json.dumps({'bundle':str(a.output),'artifacts':len(artifacts),'bytes':sum(x['byteCount'] for x in artifacts)}))
