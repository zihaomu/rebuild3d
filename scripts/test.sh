#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

developer_dir="$(xcode-select -p)"
frameworks="$developer_dir/Library/Developer/Frameworks"
runtime="$developer_dir/Library/Developer/usr/lib"
if [[ -d "$frameworks/Testing.framework" && -f "$runtime/lib_TestingInterop.dylib" ]]; then
    # Command Line Tools ship Swift Testing outside SwiftPM's default framework search path.
    swift test --disable-xctest \
        -Xswiftc -F -Xswiftc "$frameworks" \
        -Xlinker -rpath -Xlinker "$frameworks" \
        -Xlinker -rpath -Xlinker "$runtime" "$@"
else
    swift test --disable-xctest "$@"
fi
