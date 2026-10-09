# Rebuild3D v0.1.0 Mac 安装说明

适用于 **Apple Silicon（M 系列）Mac、macOS 26 或更新版本**。已在 Apple M5、16 GiB 内存、macOS 26.4.1 上验证，其他机型尚未实机验证。建议至少预留 20 GB 磁盘空间，用于应用、首次组件准备和生成结果。

## 下载与安装

1. 打开 [v0.1.0 Release](https://github.com/zihaomu/rebuild3d/releases/tag/v0.1.0)，展开 **Assets**。
2. 下载 `Rebuild3D-0.1.0-macos-arm64.dmg` 和全部同名 `.dmgpart` 文件，放在**同一文件夹**，保留原文件名。
3. 双击 `.dmg` 文件。macOS 会自动读取其他分卷，无需执行合并命令。
4. 将窗口内的 **Rebuild3D.app** 拖入旁边的 **Applications（应用程序）**。
5. 从“应用程序”打开 Rebuild3D，完成安装后可推出磁盘映像。

三个安装文件合计约 **4.99 GB**，都需要下载：

| 文件 | 大小 |
| --- | --- |
| [Rebuild3D-0.1.0-macos-arm64.dmg](https://github.com/zihaomu/rebuild3d/releases/download/v0.1.0/Rebuild3D-0.1.0-macos-arm64.dmg) | 1.99 GB |
| [Rebuild3D-0.1.0-macos-arm64.002.dmgpart](https://github.com/zihaomu/rebuild3d/releases/download/v0.1.0/Rebuild3D-0.1.0-macos-arm64.002.dmgpart) | 1.99 GB |
| [Rebuild3D-0.1.0-macos-arm64.003.dmgpart](https://github.com/zihaomu/rebuild3d/releases/download/v0.1.0/Rebuild3D-0.1.0-macos-arm64.003.dmgpart) | 1.00 GB |

GitHub [单个 Release 附件须小于 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)，因此完整离线应用采用分卷。`.dmgpart` 不是独立安装包，不能只下载其中一份。`Source code (zip)` 和 `Source code (tar.gz)` 是源码，也不是应用。

Release 同时提供 `INSTALL-macOS.txt` 和 `SHA256SUMS.txt`，分别用于离线阅读安装步骤和核对下载完整性。

## 安装包里有什么

以下为应用展开后的实际文件大小，GB/MB 使用十进制单位；应用总计约 **6.23 GB**。

| 内容 | 大小 | 用途 |
| --- | --- | --- |
| VGGT-1B 原始 FP32 权重 | 5.03 GB | 少图相机与深度预测，约占应用的 81% |
| PyTorch | 520 MB | 本地模型推理与张量计算 |
| OpenUSD（`pxr`） | 239 MB | USD 场景、材质及 USDZ 导出 |
| OpenCV（`cv2`） | 125 MB | 图像与前景掩膜处理 |
| Python 与其余依赖 | 304 MB | 包括 NumPy、SciPy、scikit-image、Pillow 等 |
| 原生程序、工作进程、清单及签名等 | 17 MB | Mac 主程序本身约 2 MB，其余主要为组件清单与签名 |

安装包没有包含用户的 `data`、`data2` 照片或验收生成结果。体积主要来自模型权重和运行依赖。

首次生成还会将约 6.22 GB 组件复制到应用管理目录，用于完整性检查、断点准备和修复。因此应用与组件副本合计约 **12.45 GB**，另需下载分卷、照片、检查点与模型结果的空间。当前包保留已验收的完整权重和依赖，尚未做权重裁剪或依赖精简。

已核对的精简机会（尚未实施）：

- 当前推理关闭 `point_head` 和 `track_head`，原始权重仍包含这两部分，合计约 **394 MB**。
- `aggregator` 权重以 FP32 保存，占 **3.64 GB**，MPS 推理加载后会转换为 FP16。若提前按相同精度保存该部分，理论上可再减少约 **1.82 GB**。结合移除未用参数，预计权重可从 5.03 GB 降至约 **2.81 GB**；这只是按张量大小计算的估计，尚未验证精简包或压缩后的下载体积，CPU 的 FP32 路径也不能据此宣称等价。
- Python 目录有三个内容相同的启动文件，额外占约 **36 MB**；另含约 **37 MB** 测试目录和 **41 MB** 头文件，可逐项检查运行依赖后精简，不能直接整批删除。
- 减少应用包与管理目录之间的模型重复存储，可以降低安装后的磁盘占用，但不会自动缩小下载包。

这些调整需要更新固定摘要和组件清单，并重新跑两套七图回归；本次 v0.1.0 完整包保持已验收内容。

## 首次打开被 macOS 阻止

当前使用 ad-hoc 签名，尚未取得 Developer ID 签名和 Apple 公证，不能保证双击后直接越过系统确认。确认下载来自本项目 Release 后：

1. 先尝试打开“应用程序”中的 Rebuild3D。
2. 前往 **系统设置 → 隐私与安全性**，找到对应 Rebuild3D 的提示，点击 **仍要打开**。
3. 在再次出现的对话框中确认打开。之后可以正常双击启动。

这是 Apple 提供的[允许打开未公证应用的操作](https://support.apple.com/zh-cn/102445)。如果设备受单位管理，该选项可能由管理员控制。

## 开始生成

点击“添加照片”，导入同一物体的照片，再点击“生成模型”。应用包已包含 Python、依赖与 VGGT-1B 权重，无需安装 Xcode、Python 或另行下载模型。首次生成会从包内准备约 6.22 GB 组件，随后可离线使用。

少图模型的几何属于推测。可在“查看来源”中检查推测区域与照片颜色来源；操作、保存、恢复及质量限制见[使用说明](v3-本地应用使用与交付说明.md)。

VGGT 代码和权重适用各自许可；随包 VGGT-1B 权重标注 **CC BY-NC 4.0，限非商业使用**。应用“高级选项 → 查看本地组件许可…”保留组件声明，源码的 Apache-2.0 许可不替代第三方许可。

## 安装包验证范围

本机检查包括磁盘映像挂载、从映像复制应用、深度签名检查、组件文件摘要、移位后的 Python 依赖与 Metal 支持，以及应用启动。重建算法与此前通过两套七图验收的发布构建一致。

这些结果不代表已完成另一台 Mac 或下载隔离状态下的全流程测试；Developer ID 签名、公证与其他硬件兼容性仍待完成。
