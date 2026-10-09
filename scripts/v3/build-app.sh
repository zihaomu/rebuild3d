#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $# -ne 2 || ! -f "$1/runtime.json" || -e "$2" ]]; then
    echo 'Usage: scripts/v3/build-app.sh RUNTIME_DIRECTORY NEW_APP_PATH' >&2
    exit 1
fi
runtime="$(cd "$1" && pwd)"
destination="$2"
swift build --product Rebuild3D --configuration release
binary_dir="$(swift build --show-bin-path --configuration release)"
mkdir -p "$(dirname "$destination")"
staging="$(mktemp -d "$(dirname "$destination")/.Rebuild3D-v3.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
app="$staging/Rebuild3D.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$binary_dir/Rebuild3D" "$app/Contents/MacOS/Rebuild3D"
cp Config/Info.plist "$app/Contents/Info.plist"
# Release version comes from Config/Info.plist; the v3 plan is not an app version.
cp LICENSE "$app/Contents/Resources/LICENSE"
cp Vendor/Photogrammetry/LICENSE "$app/Contents/Resources/Photogrammetry-MIT-LICENSE"
# APFS clones avoid another physical copy of the 5 GB checkpoint; normal copy is the fallback.
cp -cR "$runtime" "$app/Contents/Resources/GenerationRuntime" || {
    rm -rf "$app/Contents/Resources/GenerationRuntime"
    cp -R "$runtime" "$app/Contents/Resources/GenerationRuntime"
}
codesign --force --sign - "$app"
codesign --verify --strict "$app"
mv "$app" "$destination"
echo "Built $destination"
