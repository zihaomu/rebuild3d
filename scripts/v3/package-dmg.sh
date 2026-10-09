#!/bin/bash
# Package an already verified full app; each release asset stays below 2 GiB.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $# -ne 2 || ! -f "$1/Contents/Resources/GenerationRuntime/runtime.json" || -e "$2" ]]; then
    echo 'Usage: scripts/v3/package-dmg.sh FULL_APP_PATH NEW_OUTPUT_DIRECTORY' >&2
    exit 1
fi
app="$(cd "$1" && pwd)"
version="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$app/Contents/Info.plist")"
minimum_os="$(/usr/libexec/PlistBuddy -c 'Print LSMinimumSystemVersion' "$app/Contents/Info.plist")"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "$(lipo -archs "$app/Contents/MacOS/Rebuild3D")" == arm64 ]]
codesign --verify --deep --strict "$app"
mkdir -p "$2"
destination="$(cd "$2" && pwd)"
staging="$(mktemp -d "$(dirname "$destination")/.Rebuild3D-dmg.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
mkdir "$staging/volume"
ditto "$app" "$staging/volume/Rebuild3D.app"
ln -s /Applications "$staging/volume/Applications"
cat > "$staging/volume/安装说明.txt" <<EOF
Rebuild3D $version — macOS 安装与使用

要求：Apple Silicon（M 系列）Mac、macOS $minimum_os 或更新版本。
本机已验证 Apple M5、16 GiB 内存、macOS 26.4.1；其他机型尚未实机验证。
建议至少留出 20 GB 可用磁盘空间，用于应用、组件副本和生成结果。

1. 下载同一版本的 .dmg 和全部 .dmgpart 文件，放在同一文件夹。
   保留原文件名。分卷是同一个完整安装包的组成部分，都必须下载。
2. 双击 .dmg 文件；macOS 会自动读取旁边的 .dmgpart，无需合并命令。
3. 将 Rebuild3D.app 拖到旁边的 Applications（应用程序）文件夹。
4. 从“应用程序”打开 Rebuild3D。
   当前使用 ad-hoc 签名，尚未取得 Developer ID 签名和 Apple 公证。
   如果 macOS 阻止打开，先尝试打开一次，再前往：
   系统设置 → 隐私与安全性 → 找到 Rebuild3D → 仍要打开。
   只在确认应用来自本项目的 GitHub Release 后允许打开。
5. 点击“添加照片”，选择同一物体的照片，再点击“生成模型”。

包含 Python、本地依赖与 VGGT-1B 权重，无需安装开发工具或下载模型。
首次生成会从应用包准备约 6.22 GB 组件；之后可以离线使用。
少图几何属于推测，结果仍可能粗糙；在“查看来源”中检查推测和取色区域。

许可：Rebuild3D 为 Apache-2.0；Photogrammetry 的衍生部分保留 MIT 许可。
VGGT 代码与权重分别适用原始许可。随包的 Meta VGGT-1B 原始权重未修改，
revision 860abec7937da0a4c03c41d3c269c366e82abdf9，标注 CC BY-NC 4.0，限非商业使用。
模型来源：https://huggingface.co/facebook/VGGT-1B
模型卡：https://huggingface.co/facebook/VGGT-1B/blob/860abec7937da0a4c03c41d3c269c366e82abdf9/README.md
权重许可：https://creativecommons.org/licenses/by-nc/4.0/
应用内“高级选项 → 查看本地组件许可…”可查看组件声明，许可文件保留在应用包内。

项目与更新：https://github.com/zihaomu/rebuild3d
发布页面：https://github.com/zihaomu/rebuild3d/releases/tag/v$version
EOF
stem="Rebuild3D-$version-macos-arm64"
hdiutil create -fs HFS+ -volname "Rebuild3D $version" \
    -srcfolder "$staging/volume" -format UDZO -imagekey zlib-level=6 "$staging/full.dmg"
# Finder/DiskImageMounter still supports these native segments on the target macOS.
# hdiutil marks the format deprecated; revalidate mounting for future OS releases.
hdiutil segment -segmentSize 1900m -o "$destination/$stem.dmg" "$staging/full.dmg"
cp "$staging/volume/安装说明.txt" "$destination/INSTALL-macOS.txt"
(
    cd "$destination"
    for part in "$stem"*.dmg*; do
        [[ "$(stat -f %z "$part")" -lt 2147483648 ]]
    done
    shasum -a 256 "$stem"*.dmg* INSTALL-macOS.txt > SHA256SUMS.txt
)
echo "Packaged $destination"
