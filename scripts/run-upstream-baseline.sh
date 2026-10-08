#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 2 ]]; then
    echo "Usage: scripts/run-upstream-baseline.sh INPUT_DIRECTORY OUTPUT.usdz" >&2
    exit 2
fi
if [[ -e "$2" ]]; then
    echo "Output already exists; choose a new path." >&2
    exit 2
fi
mkdir -p build
xcrun swiftc -swift-version 5 -target "$(uname -m)-apple-macos26.0" \
    Vendor/Photogrammetry/Photogrammetry/Delegate/PhotogrammetryDelegate.swift \
    Vendor/Photogrammetry/Photogrammetry/Types/PhotogrammetryDelegateError.swift \
    Vendor/Photogrammetry/Photogrammetry/Extensions/PhotogrammetrySession.swift \
    scripts/UpstreamBaseline.swift -o build/upstream-baseline
/usr/bin/time -l build/upstream-baseline "$1" "$2"
