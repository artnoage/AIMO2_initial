#!/bin/bash

# This script launches multi-GPU training for the GRPO solver using DeepSpeed

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate command not found. Please install it with 'pip install accelerate'."
    exit 1
fi

# Check if deepspeed is installed
if ! command -v deepspeed &> /dev/null; then
    echo "Warning: deepspeed command not found. Will use accelerate with deepspeed backend."
    echo "For best performance, install deepspeed with 'pip install deepspeed'."
    USE_DEEPSPEED_DIRECT=false
else
    USE_DEEPSPEED_DIRECT=true
fi

# Default configuration
NUM_GPUS=4  # Default to 4 GPUs
SCRIPT_PATH="solver_ref_4b_trl.py"
DS_CONFIG_PATH="ds_config.json"

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
        --ds_config)
        DS_CONFIG_PATH="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--gpus NUM_GPUS] [--script SCRIPT_PATH] [--ds_config DS_CONFIG_PATH]"
        exit 1
        ;;
    esac
done

echo "Starting DeepSpeed multi-GPU training with $NUM_GPUS GPUs using script: $SCRIPT_PATH"
echo "Using DeepSpeed config: $DS_CONFIG_PATH"

# Check if the DeepSpeed config file exists
if [ ! -f "$DS_CONFIG_PATH" ]; then
    echo "Error: DeepSpeed config file not found at $DS_CONFIG_PATH"
    exit 1
fi

# Run with DeepSpeed directly or via accelerate
if [ "$USE_DEEPSPEED_DIRECT" = true ]; then
    echo "Using DeepSpeed directly"
    
    # Run with deepspeed
    deepspeed \
        --num_gpus=$NUM_GPUS \
        $SCRIPT_PATH \
        --deepspeed=$DS_CONFIG_PATH
else
    echo "Using accelerate with DeepSpeed backend"
    
    # Run with accelerate using deepspeed backend
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        --num_machines=1 \
        --main_process_port=29500 \
        --deepspeed_config_file=$DS_CONFIG_PATH \
        $SCRIPT_PATH
fi

echo "Training complete!"
