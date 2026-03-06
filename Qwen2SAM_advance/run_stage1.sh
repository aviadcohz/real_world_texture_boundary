#!/bin/bash
# ============================================================
# Stage 1: General Segmentation Recovery with Qwen replacing CLIP
# ============================================================
#
# Prerequisites:
#   1. Download COCO 2017:
#      python scripts/download_datasets.py --data-dir /data --download coco sam3
#
#   2. Update configs/stage1.yaml with correct data paths
#
#   3. Activate conda environment:
#      conda activate texture_boundary
#
# Usage:
#   bash run_stage1.sh
#
# ============================================================

set -e

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Config
CONFIG="configs/stage1.yaml"

echo "============================================================"
echo "Qwen2SAM_advance - Stage 1 Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "Working dir: $SCRIPT_DIR"
echo ""

# Check config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Check GPU
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f'GPU: {gpu.name} ({gpu.total_mem / 1e9:.1f} GB)')
else:
    print('WARNING: No GPU detected!')
"

# Check SAM3 is importable
python -c "import sys; sys.path.insert(0, '/home/aviad/sam3'); from sam3.model_builder import build_sam3_image_model; print('SAM3: OK')"

echo ""
echo "Starting training..."
echo "============================================================"

# Run training
PYTHONPATH="/home/aviad/sam3:/home/aviad/real_world_texture_boundary:$PYTHONPATH" \
python training/train_stage1.py --config "$CONFIG" 2>&1 | tee "logs/stage1_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Training complete!"
echo "Checkpoints: checkpoints/stage1/"
