#!/bin/bash
# QwenVL 2.5 Microservice - Start Server
# Run this inside byobu on MobaXterm:
#   bash start_server.sh
#
# The server will be accessible from other machines at:
#   http://<this-machine-ip>:8000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate texture_boundary

export PYTHONPATH="/home/aviad/real_world_texture_boundary/texture_boundary_pipeline:$PYTHONPATH"
cd /home/aviad/real_world_texture_boundary/texture_boundary_pipeline

echo "Starting QwenVL 2.5 server on 0.0.0.0:8000..."
python -m models.qwen_server --host 0.0.0.0 --port 8000 --batch-size 12
