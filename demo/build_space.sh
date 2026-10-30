#!/usr/bin/env bash
# Assemble everything the Hugging Face Space needs into build/space/.
#   bash demo/build_space.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build/space"
cd "$ROOT"

if [ ! -f data/raw/traffic_dataset.mat ]; then
  echo "Dataset not found - downloading it first..."
  python step1_download_data.py
fi

rm -rf "$OUT"
mkdir -p "$OUT/data/raw"
cp demo/app.py "$OUT/app.py"
cp demo/viz.py "$OUT/viz.py"
cp demo/README_space.md "$OUT/README.md"
cp demo/requirements_space.txt "$OUT/requirements.txt"
cp step1_download_data.py "$OUT/"
cp -r src docs figures results "$OUT/"
cp data/raw/traffic_dataset.mat "$OUT/data/raw/"
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$OUT" -name ".DS_Store" -delete
echo "Built $OUT ($(du -sh "$OUT" | cut -f1))"
echo "Next: python demo/deploy_space.py"
