# v4 实验入口

状态与验收以 [v4 Goal](../../doc/GOAL-第四阶段-v4模型量化与整体提速.md) 为准。这些是开发侧实验工具；本地候选应用已接入量化后端，既有 v0.1 发布保持原状。所有输出使用新目录；原图、v3 组件和旧成果保持不变。`build/` 下实验与权重不提交到 Git。

实施更新（2026-10-11）：正式 `rebuild3d_worker.infer` 与受管理工作进程已可读取 MLX 候选，支持 518/392 输入；A/B 已生成完整量化模型。新增 UV 空间分块实现与本地应用打包工具。旧安装应用保持原路线。候选 09 的正常使用、取消后重启续跑、离线 GUI、指定成品和独立 E2 相对审核已有通过证据。三轮实际应用共 12 次运行已完成，推理耗时降幅 A/B 为 18.87%/18.36%，未达 20%；实际等待与物理峰值有收益，但后台负载和 swap 条件仍需复验，最终交付审计未完成。详见 Goal 第 15 节末及 `ui-performance-01/summary-after-12.json`。

已完成指定应用功能验收的构建为 `runtime-candidate-11` / `delivery/Rebuild3D-v4-candidate-10.app`，后端版本 4 使用异步 block 提交及返回前同步，组件摘要为 `5d0446777eafc261caf583f386e634c013f330ad262ad345ad15a18790e82a10`。两轮各 12 次调度诊断均保留，第二轮固定空闲间隔下 A/B 相对候选 09 的主干中位数约减少 0.57 秒；不换算成相对 v3 的正式应用结论。68 项 Runtime 测试、九组预测完全一致、隔离 CPU 和包体/签名/依赖审计通过，见 `async-scheduling-integration-01/`。候选 10 已完成 A 的实际全新重建、保存重开和导出，见 `application-acceptance-06/`；A/B 生产核心取消/异常恢复通过，成品内容与前版完全一致，见 `generation-recovery-05/`。受控 518→392 重试和实际 B 取消后退出重启续跑也已通过；两组导出均通过独立 Preview 显示。补充九组数量/逆序预测、ZIP 中文路径搬移、空组件目录离线准备及实际离线应用的生成/保存重开/导出也已通过，汇总见 `candidate10-readiness-01/`。ZIP 为 1,844,126,430 B，仅保存在本地。候选 10 正式性能已完成 12/12，见 `ui-performance-02/summary-after-12.json` 和 `environment-review-01/review-decision.json`；A/B 推理降幅 19.43%/20.64%，A 未过 20%，A 点击窗口 swap 增量中位数上升，验收未通过。启动/加载诊断已形成下述候选 11；现按用户要求暂停。候选 09 和原始 12 次结果保留，没有新发布。

最新实验构建为 `runtime-candidate-12` / `delivery/Rebuild3D-v4-candidate-11.app`，组件摘要 `895f3e0dfd6013a07073934480f97079ab87c1571eae383281878aebb1344a6d`。仅新增 checked-hash 字节码和构建记录，原 21,959 个功能文件一致；展开 3,139,910,521 B。十二次启动诊断、4 项缓存行为检查、68 项 Runtime 测试及组件/依赖审计通过，18 组包内预测回归全部通过，汇总见 `bytecode-integration-01/integration-summary.json`。归档搬移、核心恢复与空组件断网测试通过；实际 A 正常流程、B 取消重启恢复见 `application-acceptance-07/`，最后一次离线新生成、带贴图显示、保存重开与导出文件校验见 `offline-ui-04/report.json`。离线导出后的窗口读取超时已保留，未冒称成功提示已观察。用户要求完成本次测试后停止，现已暂停；受控 392 完整重试、新正式性能资源复验及最终 P0 审计未完成，没有发布。

已有冻结基线为 `build/v4-quantization/baseline/`，原组件为 `build/v3-oneclick/runtime-app-03/`。执行 Python 时设置 `PYTHONDONTWRITEBYTECODE=1`，避免向冻结组件写入字节码。

本地文件清理（2026-10-11）：旧候选应用、旧组件副本和旧归档已纳入清理，以下旧版本路径作为历史实验记录保留，不能直接作为当前可执行入口。清理时保留的应用为 `delivery/Rebuild3D-v4-candidate-09.app`，组件为 `runtime-candidate-10`；此后新增候选 10，当前入口见上文，两者均位于 `build/v4-quantization/`；浮点基线 `build/v3-oneclick/delivery/Rebuild3D.app` 和 `runtime-app-03` 保留。候选 09 的搬移测试副本需要从保留的 `archive-candidate-09/Rebuild3D-v4-candidate-09.zip` 重新解压后再复测离线流程。删除明细、保留摘要及旧组件来源资料见本地 `build/cleanup-20261011/`，历史测试报告和重建成果保留。

同日第二轮还清理了恢复、资源重试及 E/E2 实验中的 10 套 `runtime-store` 组件副本，记录位于 `build/cleanup-20261011-extra/`。这些实验的项目、输入和结果保留；复跑隔离测试前需要重新准备组件存储。

| 工具 | 用途 |
| --- | --- |
| `probe-bytecode-startup.py compile / screen` | 使用候选 10 自带 Python 将相同源码预编译到独立目录，再交错比较有/无缓存的新推理进程；逐数组核对冻结预测，记录启动、加载及物理内存。使用 checked-hash、optimize=0，运行时禁止写缓存；不修改原组件，不计作实际应用性能通过 |
| `benchmark.py freeze / run` | 校验原图、组件与旧成果；监测独立新后端任务。后端时间不包含应用展示时间 |
| `repeat-full.py` | 校验完整组件，按冻结顺序执行 A/B 各三轮浮点/候选交错新任务；分阶段耗时、推理物理峰值、压力与 swap 分别记录。缺失采样不视为零，UI 单独验收 |
| `prepare-ui-performance.py`、`observe-application-run.py` | 冻结 12 个实际应用计时用新项目及交错顺序；从真实点击前开始只读采样，结合实际模型显示观察验收。观察器每两秒记录后台 CPU，可用 `--system-state-probe` 配合新版 `system-state.swift` 每秒记录 OS 热状态分类；不是 GPU 频率或温度测量。准备工具不启动重建，观察器不操作窗口 |
| `summarize-ui-performance.py`、`test-ui-performance-summary.py` | 汇总实际 UI 观察与采样，逐组计算三轮中位数；用点击/显示的时间上下界判断等待收益，保留缺失、失败和不确定样本。旧基线缺少推理子阶段计时会保留为空，不冒充零耗时；环境、压力与睡眠仍需单独审核 |
| `review-ui-performance-environment.py` | 在 12 个实际样本完成后提取供电、热警告记录、后台进程快照、应用重启身份、实际点击至可见期间的压力/swap、采样间隔和墙钟偏差；保留观察器原始统计，另报排除前后空闲时间的采样窗口。不自动判定环境通过或 Goal 完成 |
| `summarize-ui-telemetry.py` | 汇总新观察器的连续热状态与进程 CPU 记录，支持中途核查；从 CPU 排名中排除实际采到的应用进程树，并报告日志截断、进程读取失败和 swap/压力。剩余 CPU 汇总只是已列进程的下界，不自动证明环境可比或性能达标 |
| `profile-baseline-inference.py` | 用未修改的 v3 推理入口加进程内计时包装，补充验证、加载和同步网络前向时间，并逐数组核对原预测；额外同步会改变调度，只作子阶段诊断，不替代原应用三轮计时 |
| `repeat-packaged-inference.py` | 窗口不可用时，对两套冻结准备图执行 A/B 各三轮包内推理交错诊断；新进程加载、物理占用、压力与 swap 全部记录，核对重复预测一致。复用准备图且不含应用显示，只用于定位性能问题，不能勾选完整应用计时 |
| `probe-inference-scheduling.py` | 在冻结候选 09 的独立进程中剖析加载或试验调度方式，不修改应用包；验证照片身份、518/392、预测数组和像素变换。已实测加载剖析、异步 block、批量加载及输出头非阻塞复制；异步输入边界的结果与参考一致，性能仍需对照。提前导入改变计时边界，不能用于正式应用性能判定 |
| `repeat-scheduling-diagnostic.py`、`system-state.swift` | 对两套七图执行控制/实验各三轮新进程交错诊断，保留全部原始结果、预测一致性及采样内存；可固定每次运行前的空闲时间，并用编译后的 Swift 小工具记录当前热状态和低功耗设置。空闲间隔不保证相同温度，系统分类也不是连续温度或 GPU 频率测量 |
| `verify-inference-parity.py` | 复制并冻结指定 worker 源码，使用包内 Python、权重和 VGGT 对既有准备图重新推理；验证照片身份、执行策略版本、真实低位调用、像素变换和六组输出数组完全一致。复用准备图只用于正确性，不代替独立包、完整重建、应用恢复或性能验收 |
| `compare-job-content.py` | 比较两个实际保存任务的原图、准备图、预测数组、相机/来源及完整几何、UV、贴图、GLB/USDC 和 USDZ 内部资源。只按已核对原图对应照片 UUID；绝对来源路径因项目搬移单独验证。完全一致可保留之前视觉结论的内容范围，不冒称新的视觉、UI 或性能验收 |
| `prepare-offline-ui.py` | 为未修改的独立候选创建进程级断网测试启动器；先验证连接正对照及网络拒绝，再通过实际应用窗口操作。必须核对启动证据中的 PID、可执行文件和工作进程继承关系，准备成功不算 GUI 验收通过 |
| `convert-weights.py` | 流式生成精简 FP16 或直接 MPS W8/W4 文件；MPS 低位速度未达标，仅保留研究对照 |
| `benchmark-linear.py` | 实际大矩阵形状上的 FP16/MPS 真实低位内核交错计时 |
| `capture-block.py`、`benchmark-mlx-block.py` | 捕获本机照片特征，核对三个代表性 VGGT block 的 MLX 数值与时间 |
| `convert-mlx-weights.py` | 离线生成 MLX 真实 uint32 打包权重、量化参数和来源清单，逐张量回读验证 |
| `infer-mlx-experiment.py` | 隔离的 MLX 主干 + MPS 输出头推理；目前限制 518 输入及 3–12 张；没有接入生产恢复和组件协议 |
| `compare-inference.py` | 单一场景相似变换后的相机、深度和投影检查；数字不通过返回非零退出码，仍需单独审核模型和贴图 |
| `review-model.py` | 比较真实 USDZ 渲染的全部照片视角、原图遮罩与固定细节，记录轮廓及贴图覆盖；视觉结论单独记录 |
| `render-cpu-review.py` | 用 CPU 读取实际 USDZ 及内嵌贴图，生成照片视角和转台参考图；不能替代原生显示验收 |
| `diagnose-low-resolution-fusion.py` | 先确认相同有序原图与已存推理，再观察未修改融合的组件过滤前后结果；量化重放必须逐数组复现真实 392 重试产物 |
| `diagnose-component-gap.py` | 记录独立网格组件之间诊断采样段的逐视角深度、权重及轮廓约束；采样段不得作为补杆路径 |
| `probe-position-scale.py` | 隔离验证 392 的 RoPE patch 中心按 518 范围归一化，支持 MLX 与浮点对照；不修改正式源码，不构成通用修复结论 |
| `review-fusion-probe.py` | 对相同有序原图的多个融合变体生成逐照片 CPU 灰模；各自推理相机与尺寸明确记录，分数只用于诊断 |
| `finish-fusion-probe.py` | 复用明确指定的准备输入、推理和网格，重新制作 4K 照片贴图及 USDZ，并生成原生七视角/转台；不计为全新任务或性能验收 |
| `benchmark-uv.py` | 拆分图块生成与排布计时，验证空间分块后的原始面/角点映射；仅为 UV 诊断，不是新任务总时间 |
| `repeat-uv.py` | 冻结源码、网格和扩展，以新进程交错复测整网格/分块 UV，保存全部样本、中位数和重复输出一致性 |
| `audit-uv-layout.py`、`audit-uv-layout.cpp` | 按连续 UV 三角形检查面积重叠，测量不同图块边界的实际间距；有理数复核重叠样例，不用配置中的 padding 代替实测 |
| `audit-uv-distortion.py` | 在同一三维网格上按原面 ID 比较 UV 方向拉伸与像素密度，分别按面数和表面积统计，并区分学习深度与补全部分；不代替视觉验收 |
| `review-uv-local.py` | 对相同网格的指定面做遮挡定位、原生放大与缩小对照，并报告图集采样数量；合成诊断视角明确区别于照片观测，生成结果不自动判为视觉通过 |
| `test-uv-layout.py` | 用相交、接触、共边、反向绕序和网格边界等解析样例检查 UV 诊断工具 |
| `repair-uv-overlaps.py` | 保留的离线原型，将全部被诊断为重叠的面移入独立带边距小岛；生产实现已移入 `Runtime/rebuild3d_worker/uv_layout.py` |
| `repair-uv-slivers.py` | 狭长 UV 采样修复的离线原型；按几何畸变和 texel 高度选择，等距重排并保持图集预算，必须重新烘焙和审核 |
| `package-runtime.py` | 验证冻结基础组件、打包唯一量化权重与 MLX 依赖，校验独立 Python 的模块路径 |
| `build-app.py` | 从明确指定的 Swift 可执行文件与运行组件组装、签名本地候选，报告完整展开体积；此脚本不编译 Swift |
| `audit-app.py` | 核对候选包的完整组件摘要、签名和原生依赖路径；可结合包内 CPU 实测确认扩展加载顺序 |
| `probe-runtime.py` | 清空开发环境变量，使用包内 Python 的隔离模式实际运行 CPU 组件，并登记加载库路径；不执行 MLX 推理或应用 UI |
| `probe-component-install.swift` | 与生产 Core 源码一起编译，验证真实组件首次准备、损坏修复、复制取消续传与旧组件共存；不是 UI 或重建任务恢复验收 |

MLX 环境继承经过校验的 v3 依赖，新增依赖固定版本及摘要在 `requirements-mlx.lock`。创建新的开发环境时可使用：

```sh
uv venv --python build/v3-oneclick/runtime-app-03/python/bin/python3.12 \
  --system-site-packages build/v4-quantization/mlx-env-new
uv pip install --python build/v4-quantization/mlx-env-new/bin/python \
  --require-hashes -r scripts/v4/requirements-mlx.lock
```

现有权重转换示例（输出目录必须不存在）：

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python \
  scripts/v4/convert-mlx-weights.py \
  build/v4-quantization/weights/trimmed-fp16-01/vggt.safetensors \
  build/v4-quantization/weights/new-w8-heads16 \
  --repo build/v3-oneclick/runtime-app-03/vggt --bits 8 --group-size 64 --heads fp16
```

使用 `--heads fp16` 时，位置编码与深度/置信度激活仍保持 FP32。它是独立候选，需按其清单明确加载；不能把 FP16 派生权重再升回 FP32 当作原精度回退。MPS 与 MLX 的打包格式不同，禁止混用。

现有预打包候选推理示例：

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python \
  scripts/v4/infer-mlx-experiment.py \
  build/v4-quantization/baseline/B-fresh-01/job/inputs \
  build/v4-quantization/inference/B-new-probe \
  --repo build/v3-oneclick/runtime-app-03/vggt \
  --weights build/v4-quantization/weights/mlx-w8-g64-heads16-01/vggt.safetensors \
  --bits 8 --heads fp16
```

该例重用已准备的照片，仅用于推理比较，不是一次新的完整重建速度测试。输出保存权重/源码身份、逐阶段计时、物理内存与 GPU 分配的分项统计；不能将物理内存、RSS、MPS driver 和 MLX 分配相加。首次编译、进程启动、稳定运行和检查点恢复分别记录。

A/B 是已知需求回归，参与候选比较，不作为未见数据证明。E 骰子数据已冻结，配置冻结前禁止用来调参或重建。当前权重量化不拟合照片激活统计；若后续加入校准，需要另外登记 Q。

正式后端和派生格式已接入工作进程；后续交付仍需最终组合的完整几何和贴图检查、UV 重复对照、三轮完整应用交错基准、实际保存导出与离线验收。原权重的 CC-BY-NC-4.0 限制不会因量化改变。

## 正式后端与 UV 验证

完整任务可使用 `benchmark.py run`，任务清单中的 `runtime.weights` 指向预打包权重。使用已制作组件内的 Python，并将组件的 `worker` 目录作为 `--worker-root`，例如 `build/v4-quantization/runtime-candidate-05/worker`；该目录包含匹配源码的 `uv_layout.dylib`，不能直接用尚未编译原生库的裸 `Runtime` 目录代替。任务中的模型、VGGT 源码及其他组件路径也必须与本轮组件身份一致。工作进程会关闭 MPS 的 CPU 回退，验证文件摘要并记录实际低位调用。直接运行 `rebuild3d_worker.infer` 时，需在启动 Python 前设置 `PYTORCH_ENABLE_MPS_FALLBACK=0`；此路径需要 Apple GPU 可用。

`unwrap.py` 当前候选默认按 32,768 面上限进行空间分块，保留全局顶点法线，统一排入一个图集，再恢复原面顺序。`--chunk-face-budget 0` 可做同网格的整网格对照。仅改变 UV 不改变几何，仍需重新烘焙、检查贴图、来源和实际成品；不能把复用网格的 UV 诊断记为完整生成提速。

如需拆分原生图块生成与排布时间，按 [源码锁定清单](xatlas-source-lock.json) 获取源码和子模块，在新的源码目录应用 [测量补丁](xatlas-profile.patch)，再构建：

```sh
git -C build/v4-quantization/xatlas-source-0.0.11 apply "$PWD/scripts/v4/xatlas-profile.patch"
cmake -S build/v4-quantization/xatlas-source-0.0.11 \
  -B build/v4-quantization/xatlas-profile-build \
  -DCMAKE_BUILD_TYPE=Release -DSKBUILD_PROJECT_VERSION=0.0.11 \
  -DPYTHON_EXECUTABLE="$PWD/build/v4-quantization/mlx-env/bin/python"
cmake --build build/v4-quantization/xatlas-profile-build -j 2
```

已有测量源码目录已应用补丁，不能再次应用。标准 wheel 未暴露拆分接口时，运行记录中的两个子阶段为 `null`，仅报告合计，禁止编造拆分数字。不同原生构建的 UV 不能假定逐元素相同；最终必须冻结实际使用的扩展文件摘要并重验质量。工作进程将该摘要纳入检查点身份。[原生扩展许可](xatlas-NOTICES.txt)随可选组件保留。

三轮 UV 单阶段交错复测示例（输出目录必须不存在）：

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python \
  scripts/v4/repeat-uv.py build/v4-quantization/uv/new-repeated-run \
  --geometry-a build/v4-quantization/managed-backend-01/A-full/job/fusion/face-provenance.npz \
  --geometry-b build/v4-quantization/managed-backend-01/B-full/job/fusion/face-provenance.npz \
  --extension build/v4-quantization/xatlas-profile-build/src/xatlas.cpython-312-darwin.so
```

脚本以三轮共十二个独立进程运行正式 `unwrap`，包含退化 UV 修复；不复用原 UV。它会复制本轮工作进程源码和扩展，保留协议、每次日志、环境记录及重复数组对照。输入网格已经生成，因此即使 UV 稳定加速，也不能拿这份报告勾选最终整体性能验收。

CPU 参考审核先对基线与候选的工作目录分别运行 `render-cpu-review.py JOB_DIRECTORY`，再运行 `review-model.py BASELINE_JOB CANDIDATE_JOB REGIONS --protocol PROTOCOL --renders renders-cpu --output-name review-cpu`。产物包含实际 USDZ 摘要；已有输出目录禁止覆盖。人工观察单独保存 `visual-review.json`，保留 CPU 光照与原生显示不同的限制。

## UV 布局与重叠检查

UV 连续布局诊断可独立于 GPU 执行。以下使用新的输出目录；`--padding` 应读取被检查图集自己的 `report.json`，这里的 `5` 仅对应 A 的现有 4K 图集。

```sh
mkdir build/v4-quantization/new-layout-auditor
xcrun clang++ -O2 -std=c++17 scripts/v4/audit-uv-layout.cpp \
  -o build/v4-quantization/new-layout-auditor/audit-uv-layout
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/test-uv-layout.py \
  --auditor build/v4-quantization/new-layout-auditor/audit-uv-layout \
  --output build/v4-quantization/new-layout-fixtures
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/audit-uv-layout.py \
  build/v4-quantization/uv/A-chunks32768-review-01/texture/uv.npz \
  build/v4-quantization/new-A-layout-review \
  --auditor build/v4-quantization/new-layout-auditor/audit-uv-layout --padding 5
```

诊断报告包含实际 UV、工具与二进制摘要。小于 `1e-8` 平方 texel 的交集按数值容差处理；这不是放宽模型视觉验收。未找到半径内邻居的间距记录为未知下界，不能按零或无限间距验收。图块以共享 UV 顶点的连通分量识别；同一图块内的三角形重叠也会被检查，完整 mipmap 与原生渲染仍需另外验收。

逐面拉伸对照使用 `audit-uv-distortion.py GEOMETRY BEFORE_UV AFTER_UV NEW_OUTPUT`。`GEOMETRY` 为固定的 `face-provenance.npz`，两份 UV 必须完整映射回同一组原始面与角点。工具按每个三角面的正交表面坐标到图集像素坐标的变换计算两个方向的伸缩比；方向比为 1 表示该面没有方向性拉伸。输出包含全部逐面数据、按面数及表面积加权的分位数、学习深度与补全区域的分项结果。绝对像素密度使用当前场景坐标单位，不代表真实毫米；归一化密度用于观察分布。退化面不会静默忽略，统计阈值仅作诊断，不新增或替代 Goal 中的视觉通过门槛。解析样例测试入口为 `Runtime/tests/test_uv_distortion.py`。

重叠修复原型必须使用完整的 `affectedFaceIDs` 和匹配的 UV 摘要。它会增加一条带边距的图集区域，随后保持最大边 4096；原三角形的纹理形状保持，整体像素密度有小幅下降，报告中单列比例。不能把旧颜色图片缩放后当成新贴图，必须重新从原照片取色。

正式 `scale_uv.py` 已自动调用 `uv_layout.py` 和打包的 `uv_layout.dylib`，在最终图集步骤完成检测、修复和复查，再交给烘焙阶段。工作进程记录原生库摘要并使旧检查点失效，用户不需要调用上述诊断脚本。v3/v4 两个组件制作入口均会编译此库；历史冻结组件不变。开发侧单独运行时可通过 `REBUILD3D_UV_LAYOUT_LIBRARY` 指向已构建的库，正式工作进程始终使用自身组件内的文件。

局部 UV 复核使用候选 Python，并通过 `PYTHONPATH` 指向同一候选的 `worker`。`review-uv-local.py BEFORE_BUNDLE AFTER_BUNDLE NEW_OUTPUT --faces FACE_ID... --renderer RENDER_REVIEW_BINARY` 首先验证网格、法线、来源和相机不变，再重新导出旧 UV 的诊断副本、执行原生渲染。报告包含遮挡射线、可见面掩码、实际 USDZ 角点及图集像素中心覆盖，灰模差异如实保留。`uv-local-native-02/` 已发现 B 局部新增细线接缝和狭长 UV 采样不足，当前配置仍需修复复核；缩小图中不明显不能代替局部验收。

后续 `uv-sliver-repair-01/` 已完成通用修复的重新烘焙及原生局部、完整视角和 200/400 像素缩小图复核；B 新增接缝消失，既有粗糙和填充继续标记。生产算法在 `uv_sampling.py`，`scale_uv.py` 现在要求 `--geometry MATCHING_FACE_PROVENANCE_NPZ`，正式工作进程会自动提供并校验原面映射。A/B 正式 UV 与已审核原型逐字节一致，见 `uv-sliver-managed-01/`；完整 Runtime 47 项测试通过，无跳过。

包含此修复的新组件为 `runtime-candidate-06`，新应用为 `delivery/Rebuild3D-v4-candidate-05.app`（2,883,521,332 字节）。`final-combination-02/` 中 A/B 全新任务均退出 0、无检查点复用；`reviewed-content-comparison.json` 证明模型、UV、贴图和相机等与已通过原生审核的原型内容一致。候选 05 包内功能、完整性、签名及依赖检查见 `candidate-05-package-audit/`。候选 04 的归档、正常应用验收和已有计时保留为对应版本的历史证据；不能自动转为候选 05 已通过。正式性能、剩余输入和新候选应用回归继续待验收。

后续反序 HEIC 测试暴露了独立的融合问题：标量场内部为负、外部为正，旧 `marching_cubes(..., gradient_direction="ascent")` 得到朝内的面。Trimesh 的闭合组件修复掩盖了部分输入的错误，但不会替开放组件修正整体朝向。正式 `fuse.extract_surface` 现使用 `"descent"`，保留原有几何、来源标签和后续法线检查；闭合球面与开放平面测试覆盖此约定。当前 Runtime 51 项测试通过，记录在 `fusion-orientation-tests-01/`。

开放网格朝向修复对应的候选为 `delivery/Rebuild3D-v4-candidate-06.app`，使用 `runtime-candidate-07`，组件清单 SHA-256 为 `77f08fc182c508c85969ead96fa36a6d8ab35ddb6efa0d48d55bfbea6755e74d`。反序案例经修复后照片支持的有效 texel 比例由 3.6565% 恢复到 62.6602%。`reverse-normal-repair-01/visual-review.json` 保存受控重放的原生审核；`heic-variant-regression-02/reviewed-content-comparison.json` 进一步核实候选 06 全新任务的模型、贴图、UV 与已审原型内容完全一致，并按原图摘要核验新旧照片身份和相机对应。该证据不包含 UI、离线或受控性能结论。

`final-combination-03/` 已完成候选 06 的原顺序 A/B 全新任务，无派生阶段复用；两组数字门槛通过，模型、贴图、UV、来源和相机内容与已审原型完全一致，见其中的 `reviewed-content-comparison.json`。该轮与归档有并行活动，耗时只作诊断。

诊断旧融合行为时，使用旧候选包内 Python、相同工作进程和已保存的失败任务：

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=build/v4-quantization/delivery/Rebuild3D-v4-candidate-05.app/Contents/Resources/GenerationRuntime/worker \
build/v4-quantization/delivery/Rebuild3D-v4-candidate-05.app/Contents/Resources/GenerationRuntime/python/bin/python3.12 -B \
  scripts/v4/diagnose-fusion-normals.py FAILED_JOB_DIRECTORY NEW_DIAGNOSTIC_DIRECTORY
```

探针记录法线修复前后统计，并要求融合数组与原失败结果完全一致。不要用新候选代码诊断旧行为，也不要把同一推理的重放当成全新完整重建。旧候选仅用于复现；新任务回归应使用候选 06。

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/repair-uv-overlaps.py \
  build/v4-quantization/uv/A-chunks32768-review-01/uv \
  build/v4-quantization/new-A-layout-review/report.json \
  build/v4-quantization/new-A-uv-repair \
  --auditor build/v4-quantization/new-layout-auditor/audit-uv-layout
```

`uv-overlap-repair-02/` 保存当前 A/B 修复前后、统一使用法线修正导出器的比较。`export.py` 已改为先统一压缩顶点/法线/UV 再构建网格，避免 Trimesh 删除顶点时丢失原法线。真实 USDZ/GLB 法线回归入口为 `Runtime/tests/test_export_normals.py`；候选 04 已包含该修正和正式 UV 重叠修复，旧候选 03 不包含。

## 独立运行组件与本地应用

以下输出目录必须不存在；运行组件使用当前工作进程源码，最终验收前需冻结其身份。

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python \
  scripts/v4/package-runtime.py build/v4-quantization/runtime-new \
  --base-runtime build/v3-oneclick/runtime-app-03 \
  --weights build/v4-quantization/weights/mlx-w8-g64-heads16-01/vggt.safetensors \
  --xatlas-extension build/v4-quantization/xatlas-profile-build/src/xatlas.cpython-312-darwin.so

swift build --configuration release --product Rebuild3D
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python \
  scripts/v4/build-app.py build/v4-quantization/runtime-new \
  build/v4-quantization/delivery/Rebuild3D-v4-new.app \
  --binary .build/arm64-apple-macosx/release/Rebuild3D
```

包中省略未使用的 MLX 分布式命令行启动器，保留模块、动态库、版本信息与许可证；省略清单记录于 `v4-build.json`。`build-app.py` 使用单独的本地候选标识，避免混淆已安装应用。它记录输入二进制摘要、组件摘要和展开体积；签名及路径校验不替代离线启动、生成、恢复和保存导出验收。

历史候选 `build/v4-quantization/delivery/Rebuild3D-v4-candidate-04.app` 使用 `runtime-candidate-05`，展开 2,883,511,966 字节。`archive-candidate-04/archive.json` 记录 1,792,201,377 字节 ZIP 的摘要及含空格路径搬移结果。该版本原包和搬移副本的 CPU、完整性、签名与依赖检查均通过；权限恢复后 A/B 完整 GPU 任务、原生质量对照及实际应用正常生成/保存/导出已通过。后续候选 05 新增采样修复，最新候选状态见上文及文末 392 修复说明；异常恢复、独立离线、剩余输入矩阵与受控重复性能仍待完成。归档与解压时间不等于首次组件准备时间。

候选包的 CPU 功能和依赖检查示例（报告文件必须不存在）：

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python \
  scripts/v4/probe-runtime.py \
  build/v4-quantization/delivery/Rebuild3D-v4-new.app/Contents/Resources/GenerationRuntime \
  build/v4-quantization/delivery/new.cpu-probe.json
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python \
  scripts/v4/audit-app.py build/v4-quantization/delivery/Rebuild3D-v4-new.app \
  build/v4-quantization/delivery/new.dependency-audit.json \
  --runtime-probe build/v4-quantization/delivery/new.cpu-probe.json
```

CPU 检查实际使用候选中的 Python，以 `-I -B` 运行，并验证全部模块和已加载的非系统库位于组件内。内容包括 xatlas、图像读写、Torch CPU、Torchvision 原生算子、OpenUSD 和资源错误协议。MLX 只检查文件和版本，GPU 执行仍须另外验收。依赖审计保留初次未解析记录，通过上游加载器的 `LC_RPATH` 或与当前组件身份一致的实际加载证据解析，不能仅因同名库存在就认定成功。

此前受限环境无法访问 MPS，SwiftPM 沙箱启动及应用控制亦曾被拒绝；这些是历史记录。恢复后候选 04 完成了实际应用流程，40 项包内 Runtime 测试通过；后续 UV 修复使测试增至 47 项并通过。当前 GPU 和进程枚举可用；此前独立的 `cgWindowNotFound` 与锁屏记录保留，候选 06 已恢复窗口访问并完成保存重开导出，见 `application-acceptance-03/`。候选仍复用此前构建的 Swift Release 程序，不能因环境恢复或另编译验收探针就声称重新编译过应用。

实际应用的 A/B 正常流程证据保存在 `build/v4-quantization/application-acceptance-01/`。`save-reopen-export-review.json` 关联生成记录、已保存项目、来源图层观察、导出附件及移位 USDZ 的系统 Preview 显示。项目保存会由应用迁出草稿，以该报告中的当前路径为准。首次准备、诊断运行与最终受控性能样本分别记录。

导出检查入口为 `audit-application-export.py PROJECT EXPORTED_USDZ NEW_OUTPUT`，使用候选自带 Python。它校验实际导出的照片/相机身份、全部附件摘要、三种 GLB 的内嵌图像与原图集像素，并把 USDZ 复制到独立目录，检查其所有纹理引用均在包内。图集保持比例，检查 4096 长边及实际 UV 画幅，不要求正方形。报告只证明文件和内嵌资源；保存重开、视觉显示与离线能力必须另取实际证据。

```sh
PYTHONDONTWRITEBYTECODE=1 \
  build/v4-quantization/delivery/Rebuild3D-v4-candidate-04.app/Contents/Resources/GenerationRuntime/python/bin/python3.12 -B \
  scripts/v4/audit-application-export.py \
  build/v4-quantization/application-acceptance-01/A-candidate04.rebuild3d \
  build/v4-quantization/application-acceptance-01/A-export04.usdz \
  build/v4-quantization/new-A-export-audit
```

## 生产协调器的取消与失败恢复

`probe-generation-recovery.swift` 与未修改的 `Sources/Rebuild3DCore/*.swift` 一同编译，直接运行应用的 `GenerationCoordinator`。它使用指定候选包内的组件，在独立组件存储中准备并运行真实重建；完成后通过 `ProjectStore.open` 重开结果，再调用正式导出方法。

`run-generation-recovery.py` 先用 APFS 副本保护已有 A/B 项目。A 等待真实 UV 子进程启动后请求协调器取消；B 仅向本次测试拥有的推理子进程发送 SIGKILL，模拟阶段异常退出。两组随后各用新的协调器进程恢复。脚本逐项核对旧项目和原照片摘要、进程退出、检查点复用、任务身份、旧成果保留及新结果发布；已有输出目录会被拒绝，不覆盖历史证据。

```sh
mkdir -p build/v4-quantization/new-recovery-build
swiftc -parse-as-library -O -swift-version 6 \
  Sources/Rebuild3DCore/*.swift scripts/v4/probe-generation-recovery.swift \
  -o build/v4-quantization/new-recovery-build/probe-generation-recovery

PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/run-generation-recovery.py build/v4-quantization/new-recovery-run \
  --probe build/v4-quantization/new-recovery-build/probe-generation-recovery \
  --runtime build/v4-quantization/delivery/Rebuild3D-v4-candidate-05.app/Contents/Resources/GenerationRuntime \
  --project-a build/v4-quantization/application-acceptance-01/A-candidate04.rebuild3d \
  --project-b build/v4-quantization/application-acceptance-01/B-candidate04.rebuild3d
```

证据包含 `report.json`、各阶段 `process-result.json`、故障操作记录、原始事件和项目文件摘要。该入口验证生产核心流程；不能将命令行进程重新启动写成应用重启成功，也不能代替实际按钮、视口、离线、392 资源重试或性能验收。此前窗口工具返回 `cgWindowNotFound`，系统当时确认会话锁屏；与历史自动审批拒绝分别记录。候选 06 本轮已通过同一入口打开、显示来源、保存重开和导出，仍需中断恢复与离线验收。

当前 A/B 的统一结果在 `build/v4-quantization/generation-recovery-review-01/report.json`，关联 A 的 `generation-recovery-02/` 和 B 的 `generation-recovery-03/`：取消/失败后各 75 个原文件摘要不变，恢复分别复用 3/1 个有效阶段，新模型发布、核心重开及导出通过；模型内容与已审候选 05 一致。两份导出审计各核对 57 个附件及内嵌贴图。命令行故障模拟和恢复任务的耗时不计入最终性能成绩。

首轮观察器对暂时无法读取的命令行字段处理不全，失败记录保留在 `generation-recovery-01/`；当前记录缺失值及错误，不按零处理。A 的辅助断言另因内存日期小数秒与项目秒级存储不一致而退出，失败报告保留；正式核验改为按项目编码策略比较全部持久化字段。`probe-generation-recovery ... NEW_OUTPUT verify-saved` 仅重开并导出已保存结果，不重新生成，也不计作一次重建。驱动脚本可用 `--datasets B` 单独执行 B 故障场景；报告明确列出实际覆盖的数据集。

## 受控 MPS 分配上限与 392 自动重试

正式协调器会清理工作进程环境，因此仅在外部 shell 设置 MPS 上限不能触发该测试。`build-resource-retry-probe.py` 将生产 Swift 核心复制到独立目录，只在启动工作进程时增加两个环境变量的传递；重试判断、次数、身份、取消、发布逻辑及包内工作进程均不改动。`build.json` 保存全部原文件与编译副本摘要、唯一插入文本和二进制摘要。这是明确披露差异的测试探针，不是正式应用二进制。

[PyTorch 2.14 的 MPS 环境变量文档](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html) 定义了按设备推荐内存设置的分配硬上限及回收软上限。测试仅限制其拥有的工作进程，不消耗大量背景内存来挤压整机，也不关闭保护上限。该方法可触发真实 MPS 分配异常，但不等于整机物理内存不足或 MLX 分配器异常的验收。

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/build-resource-retry-probe.py build/v4-quantization/new-resource-probe
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/run-resource-retry.py build/v4-quantization/new-resource-retry \
  --probe-build build/v4-quantization/new-resource-probe \
  --runtime build/v4-quantization/delivery/Rebuild3D-v4-candidate-06.app/Contents/Resources/GenerationRuntime \
  --runtime-store build/v4-quantization/new-resource-runtime-store \
  --source-project build/v4-quantization/application-acceptance-01/B-candidate04.rebuild3d \
  --high-ratio 0.215 --low-ratio 0.20
```

上限比例属于故障场景设置，不是建议用户采用的生产配置。输出必须为新目录；实际结果决定是否触发 518 失败及唯一一次 392 重试。若未触发，或两次均受限，报告分别记为 `limit-did-not-trigger-retry` / `both-attempts-resource-limited`，不能记为成功恢复。通过需要真实分配错误、自动重试期间旧成果未变、392 全流程完成、相机/照片/投影一致、核心重开和导出审计。此项仍不代替实际 GUI、离线、整机压力或性能验收。

可加 `--stop-if-untriggered`：若 518 推理已成功，经协调器的取消接口停止剩余后处理，保留未触发报告与旧成果。当前 `resource-retry-01/` 的 0.215/0.20 未触发；`resource-retry-02/` 的 0.18/0.16 触发真实 518 失败及唯一一次 392 重试，但两次均受限，旧成果和进程回收检查通过。`resource-retry-03/` 的 0.20/0.18 也未触发，提前取消后旧成果不变；`resource-retry-04/` 在 0.195/0.18 下已通过：真实 518 MPS 分配失败触发唯一一次 392 重试，完成七图、256 融合、4096 贴图、核心重开和导出；旧结果保护、原项目不变、坐标核验通过。协调器及工具均退出 0，原生审核 `resource-retry-review-01/visual-review.json` 发现伞与部分环形结构缺失，392 成品质量明确未通过；流程成功不能代替质量通过。需要相同准备输入的浮点 392 对照和逐阶段定位，暂不归因为量化单独造成。该受控证据不代表整机低内存、MLX 分配器特定故障或正式应用验收。

后续隔离诊断 `low-resolution-diagnostic-01/` 已确认原始浮点与量化 392 均缺失主要部件。`position-scale-probe-01/` 只改变低分辨率 RoPE 坐标，恢复环形结构；`low-resolution-prototype-01/` 再以多视角深度支持保留独立伞体，未按物体设置特例，也未造连接。相同新规则的浮点/量化数字比较通过；`B-392-texture-pair/W8/review-v4-02/` 已审核真实原生贴图、七视角、十二角度及六处细节。主要部件恢复，但细连接缺口和局部接缝仍在，原型未进入正式源码或候选 06。具体证据与剩余验收见 Goal 第 15 节；保留旧失败记录，不把灰模改善或数字通过写成最终质量通过。

`probe-position-scale.py DATASET NEW_INFERENCE --repo REPO --weights WEIGHTS --backend mlx|floating --count 7` 用于 392 隔离实验，张数接受 3–12 且须与实际输入一致。需设置明确的工作进程 `PYTHONPATH` 与 `PYTORCH_ENABLE_MPS_FALLBACK=0`。`finish-fusion-probe.py SOURCE_JOB NEW_OUTPUT --worker-root WORKER --renderer NATIVE_RENDERER --variant LABEL INFERENCE FUSION` 可重复提供变体参数，会记录复用输入摘要，重新烘焙、补齐 UV 面积审计、导出及原生渲染。输出仍需用 `review-model.py` 比较并单独记录实际视觉结论；复用输入的实验不计入完整任务性能。

后续 `low-resolution-prototype-03/` 将近表面融合限定在 392，并恢复原碎片过滤规则。B 的伞杆与主体重新连通，A/B 的 518 融合数组与原候选逐元素相同；完整原型审核及实现复现证据分别在该目录与 `low-resolution-integration-01/`。源码已经加入上述修复，`backendVersion` 为 2，推理和融合记录各自策略。58 项 Runtime 测试通过；A/B/D12 的同规则 392 浮点/量化原生审核均通过，既有几何和贴图局限保留。候选 06 仍是旧源码；修复已进入 `runtime-candidate-08` 及 `delivery/Rebuild3D-v4-candidate-07.app`，组件清单 SHA-256 为 `8726b7b6bb609c7cee67ba063874ae1daa5a2c4818b92a35f79b397382a67a09`。新包的源码绑定、隔离 CPU、签名、完整性及依赖检查见 `candidate07-audit-01/`；`resource-retry-05/` 已完成实际受控重试、核心重开导出和旧结果保护；`resource-retry-review-02/` 对不同 UV 构建产生的实际贴图重新完成原生审核，保留局部接缝。完整 GUI、离线和性能尚未验收。

## D 十二图与 C 常规路线回归

`run-input-regression.py` 使用上述生产协调器探针，先复制已有 D12、C33 JPEG、C59 HEIC 项目，再顺序生成新任务。它要求 D 使用全部十二图的 MLX 少图路线，C 保持 Object Capture；校验原照片摘要、原成果保护、新模型发布、核心重开及导出。输出目录必须不存在，输入项目默认位于 `build/v3-oneclick/delivery/`。

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/run-input-regression.py build/v4-quantization/new-input-regression \
  --probe build/v4-quantization/generation-recovery-build-02/probe-generation-recovery \
  --runtime build/v4-quantization/delivery/Rebuild3D-v4-candidate-05.app/Contents/Resources/GenerationRuntime \
  --runtime-store build/v4-quantization/new-input-runtime-store

PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/review-input-regression.py build/v4-quantization/new-input-regression \
  build/v4-quantization/new-input-review \
  --runtime build/v4-quantization/delivery/Rebuild3D-v4-candidate-05.app/Contents/Resources/GenerationRuntime \
  --renderer build/v4-quantization/final-combination-01/render-review
```

审核工具为 D 生成全部十二个照片视角、十二个转台和六处固定细节，并运行 UV 与模型数字检查；C 生成十二角度的颜色、灰模和光照对照。C 的转台没有原照片相机对应关系。生成图片不会自动通过视觉验收，必须实际查看并记录结论。D 的旧模型仅在独立审核目录重新导出以保留法线，原项目不修改。

少图来源附件使用 `audit-application-export.py`；Object Capture 结果使用 `audit-conventional-export.py PROJECT EXPORTED_USDZ NEW_OUTPUT`，以候选 Python 运行。后者核对照片身份、输入快照和模型摘要，把 USDZ 单独复制到新目录，检查包内图片、几何和资源引用，不能替代应用显示或离线检查。

项目保存后，旧任务的 `dataset.json` 可能仍含草稿路径。`compare-inference.py REFERENCE_INFERENCE CANDIDATE_INFERENCE NEW_REPORT --protocol PROTOCOL --relocated-inputs` 仅在显式开启时允许路径变化：核对双方清单摘要、照片身份与顺序，仅排除 `sourcePath` 后比较其余元数据，并逐文件比较工作图和遮罩摘要。默认模式仍严格要求清单摘要一致；预测图像、像素变换和原数值门槛不变。五项相关测试见 `Runtime/tests/test_inference_comparison.py`，不能用路径选项放过不同输入。

当前三组核心结果位于 `input-regression-01/`，最终原生审核在 `input-review-02/visual-review.json`：D 数字门槛通过，三组已审视角未见明显新增重大退化；D 顶部、孔腔及 C 的细节与不可见区域仍有限制。首轮审核暂存文件缺失的失败记录保留在 `input-review-01/`。这些证据不覆盖候选 05 的实际按钮、应用重启、独立离线、其余 3–12 张输入变体或正式性能。E 当时尚未运行，最新首次测试结果见下文。

## 输入数量、路径、方向与顺序

`probe-input-project.swift` 与生产核心一起编译，实际调用 `PhotoImporter`、`ProjectStore.save/open`；输入 JSON 的 `sources` 按顺序包含 `path`、`sha256`，导入问题会导致测试失败。`run-input-matrix.py` 从已登记 D12 以 `floor(i * 12 / n)` 选取每个数量的视角，默认覆盖 3–11 张，`--reverse-seven` 额外测试七图反序。所有项目、路径及文件名包含中文和空格，复制的照片字节保持不变；D12 的既有结果单独引用。

候选 06 的 `input-matrix-02/` 显式指定 `--counts 3 4 5 6 7 8 9 10 11 12 --reverse-seven`，在新项目中覆盖全部数量。运行中的报告不代表整批通过；按每组退出结果和独立原生审核记录判断。

```sh
mkdir -p build/v4-quantization/new-input-probe
swiftc -parse-as-library -O -swift-version 6 \
  Sources/Rebuild3DCore/*.swift scripts/v4/probe-input-project.swift \
  -o build/v4-quantization/new-input-probe/probe-input-project

PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/run-input-matrix.py build/v4-quantization/new-input-matrix \
  --import-probe build/v4-quantization/new-input-probe/probe-input-project \
  --generation-probe build/v4-quantization/generation-recovery-build-02/probe-generation-recovery \
  --runtime build/v4-quantization/delivery/Rebuild3D-v4-candidate-06.app/Contents/Resources/GenerationRuntime \
  --runtime-store build/v4-quantization/new-input-runtime-store \
  --counts 3 4 5 6 7 8 9 10 11 12 --reverse-seven
```

每组都是新的 518/256/4096 任务，核对全部照片、相机和来源身份，实际重开导出并检查 USDZ/GLB 的内嵌资源；对每张图片另按独立公式验证 EXIF 变换、半像素缩放、画幅、内参与投影坐标链。资源不足导致重试会被单独暴露，不能当作固定 518 成功。原图与源项目保持不变。所有进程命令、退出码、源码副本及摘要均保留。

`create-orientation-fixtures.swift` 同样与生产核心一起编译，参数为 `REGISTERED_D_PROJECT NEW_OUTPUT`。它生成八张受控 JPEG：逆向旋转或镜像像素，再写入 EXIF 1–8，同时保存预期正向 PNG 与实际原生解码 PNG。副本为 1600 像素 SDR，不是原相机照片或 HDR 测试。预先登记的 JPEG 像素误差门槛为平均绝对 RGB 误差 ≤3/255、P99 ≤20/255。

`run-input-variant.py INPUT_LIST_JSON NEW_OUTPUT` 使用与上例相同的四个探针/组件参数，运行一个明确登记的输入变体。若提供方向副本清单，先核对压缩容差，再要求生产准备输出的全图与已核验的正向解码图像逐像素相等。清单可以提供 `expectedTypeIdentifier`，检查扩展名不符时仍按真实内容导入；此项不会重编码原照片。输入的原始来源和派生文件摘要分别保存。

`render-input-matrix.py RUN_REPORT NEW_OUTPUT --renderer RENDER_REVIEW_BINARY` 为已完成的任务生成全部照片视角与十二角度转台，以及原图/贴图/灰模组合图。可用 `--names D03 ...` 显式选择正在运行矩阵中的已完成用例；工具保存当时的报告快照，不把子集当作整个矩阵。每份渲染报告的视觉结论仍为 pending，实际查看后另写视觉记录。这些入口都不代替 UI、离线或受控性能验收。

已有原生审核可通过 `compare-reviewed-inputs.py OLD_RUN NEW_RUN NEW_OUTPUT --reviews REVIEW_JSON... --names D03...` 绑定到新候选，但必须证明内容相同。工具逐张核对原图和准备图摘要、照片身份对应，要求模型、UV、贴图、来源文件及 USDZ 全部成员内容一致，相机与来源元数据在身份对应后也须一致；旧审核图片和模型摘要同样核实。差异会阻止复用审核，转入新成品查看；成功只覆盖显式选择的已完成用例，并保留旧审核记录的全部限制。

## 当前候选的本地 ZIP 与搬移检查

```sh
PYTHONDONTWRITEBYTECODE=1 build/v4-quantization/mlx-env/bin/python -B \
  scripts/v4/archive-candidate.py \
  build/v4-quantization/delivery/Rebuild3D-v4-candidate-06.app \
  build/v4-quantization/new-candidate-archive
```

输出目录必须不存在。工具创建 ZIP、检查 CRC 和成员路径，在含中文与空格的位置解压，再调用 `probe-runtime.py` 和 `audit-app.py` 核查包内组件、签名及依赖。`archive.json` 保存进程结果、ZIP/组件/程序摘要和逻辑大小。归档耗时不是首次准备或生成速度；逻辑大小不是 APFS 独占物理占用或升级峰值，归档搬移成功也不能替代独立离线运行。

候选 06 已完成上述检查，记录为 `archive-candidate-06/archive.json`：ZIP 1,792,205,604 字节，SHA-256 `6513614449b91ef8bef87ef4face8259608cb69c99e7234c2ce3634848ad6822`；原包及搬移副本均为 2,883,521,896 字节。该产物只在本地生成，未发布。

## 配置冻结后的 E 保留对象

`run-heldout.py REGISTRY NEW_OUTPUT --runtime CANDIDATE_RUNTIME --baseline-runtime FROZEN_V3 --import-probe IMPORT_PROBE --generation-probe GENERATION_PROBE` 要求登记表的 `firstInferenceRun` 为空，校验源码、探针与输入摘要，在首次推理前写入配置冻结记录，再由生产导入器和协调器分别执行浮点及候选新任务。配置为 518/256/4096，七张全部参与；原照片保持不变。首次尝试在登记表中保留，失败不会静默重置为从未运行。质量、实际 UI 和性能另行验收；若根据 E 修改配置，必须另补独立保留对象。当前执行目录为 `heldout-final-01/`。

`review-heldout.py RUN NEW_OUTPUT --renderer NATIVE_RENDERER --protocol PROTOCOL` 对 E 的完整新任务补齐 UV 审计、原生照片视角及转台，比较推理和成品数字。所有照片的完整画面都作为复查区域，不按模型结果挑选。视觉结论须实际查看后另记。

## 强制离线的核心流程验收

`run-offline-core.py NEW_OUTPUT --app PACKAGED_APP --source-project SOURCE_PROJECT --import-probe IMPORT_PROBE --generation-probe GENERATION_PROBE` 使用生产导入器创建无缓存的新项目。隔离策略只作用于本次协调器和后代，禁止网络及开发 Python、Runtime 源码、研究权重目录读取，不修改全机网络。先以本机临时监听端口证明有无限制时的连接差异，再执行真实打包组件、核心重开和导出。记录新组件存储的逻辑字节采样；APFS 共享块不当成独占磁盘占用，采样峰值不是精确瞬时峰值。该入口不代替实际应用 UI。`offline-preflight-01/` 已证明策略生效且包内 GPU 小计算仍可用；完整任务另行记录。

候选 07 本地 ZIP 及中文路径搬移证据为 `archive-candidate-07/archive.json`：ZIP 1,792,206,463 字节，SHA-256 `97c4819f1b8a6f0e4c64101cc9f477b167ae35ead6596ca97624c73f2c2425ef`；展开 2,883,524,533 字节。没有上传或发布。

E 首次任务 `heldout-final-01/` 因旧工具的 600 秒总时限在浮点 UV 阶段终止，推理与融合已完成，原失败记录保留。`heldout-continuation-01/` 使用 1800 秒上限，按原配置恢复同一浮点任务、再运行全新候选任务；不是重新选样或调参，也不计作新任务性能样本。`run-heldout.py --phase-timeout` 可明确设定质量任务等待上限，默认 1800 秒；生产流程及质量门槛不变。

`offline-core-01/report.json` 已终态通过：候选 07 的 ZIP 搬移副本在空组件存储、强制断网及开发目录拒绝访问的条件下，完成 A 七图的新任务、核心重开及导出，原项目未变。应用加新组件存储的最终逻辑字节为 5,756,183,435，不含 ZIP 和项目；并行执行不作为正式性能样本，实际 GUI 断网另行验收。

`heldout-continuation-01/` 的浮点恢复与候选新任务均已生成成品，但 `heldout-review-01/` 的质量检查未通过：旋转 P95 2.1374°，照片 texel 覆盖下降 2.900 个百分点，六个视角轮廓超限。实际七视角及十二角度审核记录在 `candidate/review-v4/visual-review.json`，两种成品均有形变和缺损。原失败、恢复、数字和视觉报告分别保留，E 登记表已关联质量失败；不降低门槛，若依据 E 修改算法/配置，另补独立保留对象。

## 后端数值差异诊断

`trace-inference-backend.py DATASET NEW_OUTPUT --repo VGGT --weights VERIFIED_WEIGHTS --backend torch|mlx-float|mlx-w8` 在不改变推理数学的前提下保存每层预定位置的特征样本、相机头输入与四次迭代输出；MLX 诊断使用 FP32 输出头。采样增加同步和磁盘写入，不能用来比较速度。`heldout-layer-trace-01/comparison.json` 已核对三组插入观察后的最终输出分别与未观察运行完全相同，且原 MPS 重跑等于冻结参考。

`probe-backend-arithmetic.py DATASET NEW_OUTPUT --repo VGGT --weights VERIFIED_WEIGHTS --mode normalization|frontend|rope|frontend-rope` 隔离图像归一化、含 bias 的块卷积和 RoPE 表的后端运算差异。默认浮点主干及 FP32 输出头；`--bits 8 --heads fp16` 可显式使用对应打包权重。`--dino-prefix 8|16 --floating-weights VERIFIED_TRIMMED_WEIGHTS` 只在 MPS frontend 模式下执行连续的原版浮点 DINO 前缀，并记录剩余低位内核调用及参数数目。它使用两份研究检查点，未实现独立混合权重打包，不能冒充正式应用、实际分发体积或性能达标。每次保存实际生成的函数、脚本副本与调用计数；所有质量门槛保持不变。

E 已用于诊断，独立替补 E2 的输入身份及原许可见 `doc/v4-新增保留测试输入清单-E2.json`。E2 尚未运行；新生产配置冻结前不得用于精度选择。

`probe-mlx-linears.py DATASET NEW_OUTPUT --repo VGGT --weights VERIFIED_W8_WEIGHTS --heads fp16|fp32` 将打包 MLX Linear 放入原 MPS 主干算式，其余数学不迁移。它通过明确的 CPU 内存副本在两个 GPU 框架间传递数组，模型矩阵计算仍在 GPU；性能和内存须另测。可显式加 `--mode dequantized-reference` 做权重误差对照：还原 FP16 权重后使用 MPS 浮点 Linear，真实低位内核调用应为零，绝不能以此模式作为量化达标证据。

`probe-linear-precision.py DATASET NEW_OUTPUT --repo VGGT --weights VERIFIED_WEIGHTS --group-size 32|64 --policy promoted-factors|recomputed-factors` 对照 packed W8 的精度与舍入策略。`promoted-factors` 接受对应分组的离线 FP16 因子权重，保留整数打包值，只将因子提升为 FP32；`recomputed-factors` 接受经过验证的 trimmed-fp16 源权重，在研究加载阶段使用 FP32 算法重新打包。两者都执行 FP32 输入/因子/结果的真实低位矩阵计算，在 FP32 中加 bias 后统一转回 FP16。输出头默认保持既有 FP16 策略；`--heads fp32` 配合相应 FP32 头权重可做独立诊断，不能把 FP16 升型冒充原 FP32 参数。工具保存源码副本、调用覆盖和实际存储统计。研究时在线量化不构成可交付格式，FP32 运算也不能预先认定提速，必须另过完整质量及性能门槛。预登记对照位于 `heldout-linear-precision-01/`。

该对照的 group 32 两组在 E 推理门槛通过，group 64 提升运算精度对照未通过。正式转换器新增 `--linear-compute-policy fp32-packed-gemm-bias-fp16-output-v1`，配合 `--bits 8 --group-size 32 --heads fp16` 将策略绑定到嵌入式和伴随权重清单。文件继续保存 uint32 打包整数及 FP16 因子；正式加载时只提升小型因子，执行 FP32 低位内核，在 bias 后转回 FP16，不现场量化或还原整块浮点权重。未声明新策略的原 group 64 文件保持旧数学；其他组合拒绝。

本地候选 08 使用 `runtime-candidate-09`，后端实现版本 3，运行组件清单 SHA-256 为 `70b4f1fd0afe60d9e9b8acd3dc06f30f798a76e8b3988876266f30de70e8ab50`。61 项 Runtime 测试及 A/B 518、A/B/D12 392 正式推理回归通过，E 正式完整任务与原生数字检查通过；记录见 `linear-precision-integration-01/`。E 灰模局部新增微小暗点/间隙仍需诊断，视觉未判通过；原参考本身也有明显形变。候选只在本地生成，应用完整回归、独立 E2、离线与三轮性能均未由这些结果替代。

## 候选 09 的表面间隙修复

`surface_smoothing.py` 在融合场由照片轮廓约束主导的区域固定至少 45° 的锐边顶点，同时保护拓扑边界；其余表面保持原有三次 Taubin 平滑。使用场值相等的数值容差，不使用对象标签或局部人工遮罩，不填孔或新增面。来源判定在变形后重新计算。

对应 `runtime-candidate-10`、`delivery/Rebuild3D-v4-candidate-09.app`，组件 SHA-256 `1796bc7e6bff7fc2c59397fcef304268b2cf71c11915b54169425846cfb95069`。66 项 Runtime 测试通过。`E-gap-repair-02/acceptance-summary.json` 汇总 E/A/B 518 和 A/B/D12 392 的完整下游对照；这些复用推理，不能算新任务性能。B392 首次实验输入 ID 不匹配的失败保留，独立 continuation 使用正确来源；`finish-fusion-probe.py` 已在 UV 前提前检查 ID。

E 固定区域的原生暗孔像素由候选 08 的 29 降至 2，浮点参考为 7；少量穿透孔、细小凹痕和已有形变仍在。此结果是局部相对修复，不证明真实几何精度。既有 B392 甲片亮缝保留。E 已参与诊断，独立 E2 清单仍单独维护。

`run-heldout.py` 接受登记的 E/E2 编号，冻结配置并登记首次尝试后执行新核心任务；失败不能清空登记重新当作首次。`review-heldout.py --candidate-label W8-candidate09` 生成全部原生视角和数字报告，视觉结论仍单列。

完整三轮后端测量入口（与其他重建、归档、测试任务串行运行）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  build/v4-quantization/mlx-env/bin/python -B scripts/v4/repeat-full.py \
  build/v4-quantization/new-full-performance \
  --baseline build/v3-oneclick/runtime-app-03 \
  --candidate build/v4-quantization/delivery/Rebuild3D-v4-candidate-09.app/Contents/Resources/GenerationRuntime
```

这十二次完整新任务不包含 UI 点击、组件首次准备及模型显示；不能把后端总时间直接称为用户等待时间。所有样本和环境记录保留；数字门槛与同条件环境复核分别判定。

候选 09 后续证据：`generation-recovery-04/` 完成生产核心 UV 取消与推理异常后的恢复；`resource-retry-06/` 实际触发一次 518→392 资源重试，`resource-retry-review-03/` 保存其新贴图原生复核；`offline-core-02/` 完成进程级断网及禁读开发组件条件下的完整新任务、重开与导出。以上不等同于 GUI 断网或应用重启恢复。

E2 的 `heldout-E2-final-01/` 首次旧版任务在 835,492 面的 UV 阶段超过 1800 秒总时限，原失败保留。`heldout-E2-continuation-01/` 使用原冻结配置，先运行全新候选，再恢复同一基线任务，质量等待上限 7200 秒；原推理与融合摘要受保护，恢复耗时不能混入新任务性能。最终数字与原生成品审核另记，运行中不算通过。

候选 09 后续实际窗口记录在 `application-acceptance-05/save-reopen-export-review.json`：A 已保存项目重开、颜色来源标记、B 结果打开/保存/导出，以及 A/B 隔离 USDZ 在系统 Preview 中的贴图显示已确认。取消并退出应用后的阶段恢复、GUI 离线与正式重复性能分别记录，不由导出检查代替。


## 构建期字节码实验

`precompile-runtime.py NEW_RUNTIME` 必须用该新组件自带的 Python 以 `-I -B` 执行。工具拒绝已存在 `runtime.json` 的冻结目录，保留原源码，生成 optimize=0 的 checked-hash 缓存与 `bytecode-manifest.json`；不写入开发虚拟环境或已有应用。依据 [CPython 的缓存失效规则](https://docs.python.org/3.12/library/py_compile.html#py_compile.PycInvalidationMode.CHECKED_HASH)，导入时仍检查源内容。最终组件清单再对源码、缓存及其记录逐文件计算 SHA-256。

`package-runtime.py` 新增可选 `--precompile-bytecode`，在新运行组件封存前执行上述步骤；从基组件复制时不继承旧字节码。对应检查入口为 `test-runtime-bytecode.py`，覆盖中文/空格路径搬移、实际读取缓存、相同时间戳和长度的源码变更失效、断言与文档字符串保留、构建路径可复现、拒绝修改冻结目录和越界文件。

当前隔离实验为 `bytecode-cache-prototype-01/`、`bytecode-startup-screen-01/` 和 `bytecode-startup-repeated-01/`。外部缓存目录用于研究，不能当作可分发组件；正式构建、完整性、功能及实际应用性能仍需分别验证。新缓存不包含照片或重建结果，也不放宽原权重与源码校验。

`verify-inference-parity.py --packaged-worker` 可在指定组件的实际 worker 目录直接执行，使用其缓存；不创建源码覆盖层。仍要求传入同一组件的 `--worker-root`。候选 11 的 18 组检查全部通过；这属于正确性回归，不计作新任务或 UI 性能通过。
