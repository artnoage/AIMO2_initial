#!/bin/bash

# This script launches multi-GPU training for the GRPO solver

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate command not found. Please install it with 'pip install accelerate'."
    exit 1
fi

# Default configuration
NUM_GPUS=4  # Default to 4 GPUs
SCRIPT_PATH="solver_ref_4b_trl.py"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpus)
        NUM_GPUS="$2"
        shift
        shift
        ;;
        --script)
        SCRIPT_PATH="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--gpus NUM_GPUS] [--script SCRIPT_PATH]"
        exit 1
        ;;
    esac
done

echo "Starting multi-GPU training with $NUM_GPUS GPUs using script: $SCRIPT_PATH"

# Run with accelerate
accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --num_machines=1 \
    --main_process_port=29500 \
    $SCRIPT_PATH

echo "Training complete!"
