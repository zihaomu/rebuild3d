#!/bin/bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: run-sparse-pipeline.sh DATASET NEW_OUTPUT VGGT_REPO WEIGHTS PYTHON"
    exit 2
fi
sparse_dataset="$1"
sparse_output="$2"
sparse_repo="$3"
sparse_weights="$4"
sparse_python="$5"
sparse_scripts="$(cd "$(dirname "$0")" && pwd)"
[[ ! -e "$sparse_output" ]] || { echo "Output must be new"; exit 2; }
mkdir -p "$sparse_output"
"$sparse_python" "$sparse_scripts/sparse/run_vggt.py" "$sparse_dataset" "$sparse_output/inference" --repo "$sparse_repo" --weights "$sparse_weights" --count 7 --size 518
"$sparse_python" "$sparse_scripts/sparse/export_depth.py" "$sparse_output/inference"
"$sparse_python" "$sparse_scripts/sparse/fuse_depth.py" "$sparse_output/inference" "$sparse_output/fusion"
"$sparse_python" "$sparse_scripts/sparse/export_usdz.py" "$sparse_output/fusion" --output "$sparse_output/fusion/usdz"
swift "$sparse_scripts/render-sparse-model.swift" "$sparse_output/fusion/usdz/model.usdz" "$sparse_output/fusion/renders" "$sparse_output/inference/cameras.json" "$sparse_output/fusion/report.json"
"$sparse_python" "$sparse_scripts/sparse/assemble_review.py" "$sparse_output/inference" "$sparse_output/fusion"
"$sparse_python" "$sparse_scripts/sparse/create_bundle.py" "$sparse_dataset" "$sparse_output/inference" "$sparse_output/fusion" "$sparse_output/fusion/usdz" "$sparse_output/result"
