#!/bin/bash
# Download Qwen2.5-0.5B-Instruct from HuggingFace Hub to the standard models dir.
# Run on a LOGIN NODE (compute nodes typically lack internet access):
#   bash marc_files/sweep_optimizers/download_qwen_instruct.sh

set -e

MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR="/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models/Qwen2.5-0.5B-Instruct"

if [ -d "$OUT_DIR" ] && [ -f "$OUT_DIR/model.safetensors" ]; then
    echo "Already downloaded: $OUT_DIR"
    exit 0
fi

mkdir -p "$OUT_DIR"

module load Miniforge3/26.1.0-fasrc01 2>/dev/null || true
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh 2>/dev/null || true
conda activate eoss || true

echo "Downloading $MODEL_ID -> $OUT_DIR"
python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$OUT_DIR",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print("Done:", "$OUT_DIR")
EOF

chmod -R 750 "$OUT_DIR"
echo "Download complete: $OUT_DIR"
