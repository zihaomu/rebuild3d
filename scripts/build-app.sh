#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

configuration="${1:-debug}"
if [[ "$configuration" != debug && "$configuration" != release ]]; then
    echo "Usage: scripts/build-app.sh [debug|release]" >&2
    exit 1
fi
swift build --product Rebuild3D --configuration "$configuration"
binary_dir="$(swift build --show-bin-path --configuration "$configuration")"
mkdir -p build
staging="$(mktemp -d "$PWD/build/.Rebuild3D.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
app="$staging/Rebuild3D.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$binary_dir/Rebuild3D" "$app/Contents/MacOS/Rebuild3D"
cp Config/Info.plist "$app/Contents/Info.plist"
cp LICENSE "$app/Contents/Resources/LICENSE"
cp Vendor/Photogrammetry/LICENSE "$app/Contents/Resources/Photogrammetry-MIT-LICENSE"
codesign --force --sign - "$app"
codesign --verify --strict "$app"
rm -rf build/Rebuild3D.app
mv "$app" build/Rebuild3D.app
echo "Built $PWD/build/Rebuild3D.app"
echo "Run: open build/Rebuild3D.app"
