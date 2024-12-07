import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
import argparse

def setup_model(model_path="artnoage/metastral"):
    """Initialize the base model with LoRA configuration"""
    try:
        # First try loading from local path
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Could not load from local path, trying HuggingFace Hub: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            "artnoage/metastral",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "artnoage/metastral",
            trust_remote_code=True
        )
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='Export trained model with LoRA adapters')
    parser.add_argument('--checkpoint_dir', type=str, default='train_results',
                      help='Directory containing the training checkpoints')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save the exported model')
    parser.add_argument('--model_path', type=str, default='artnoage/metastral',
                      help='Path to the base model weights')
    args = parser.parse_args()

    # Setup logging
    logging.set_verbosity_info()
    
    # Initialize model and tokenizer
    model, tokenizer = setup_model(args.model_path)
    
    # If no specific directory provided, find the latest timestamp directory
    base_dir = args.checkpoint_dir
    if os.path.samefile(base_dir, 'train_results'):
        # Get all timestamp directories
        timestamp_dirs = [d for d in os.listdir(base_dir) 
                        if os.path.isdir(os.path.join(base_dir, d)) 
                        and d[0].isdigit()]  # Timestamps start with digits
        if not timestamp_dirs:
            raise ValueError(f"No timestamp directories found in {base_dir}")
        # Sort by timestamp (assuming YYYYMMDD_HHMMSS format)
        latest_dir = sorted(timestamp_dirs)[-1]
        checkpoint_dir = os.path.join(base_dir, latest_dir)
        print(f"Using latest timestamp directory: {latest_dir}")
    else:
        checkpoint_dir = base_dir

    # Find the latest checkpoint in the selected directory
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        # If no checkpoint folders found, try the main directory
        if os.path.exists(os.path.join(checkpoint_dir, 'pytorch_model.bin')):
            checkpoint_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        else:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    else:
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        if not os.path.exists(os.path.join(checkpoint_path, 'adapter_model.safetensors')):
            raise ValueError(f"Model weights not found at {checkpoint_path}/adapter_model.safetensors")
    
    # Load the adapter in float16
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use the timestamp from training directory for output
    output_dir = os.path.join(args.output_dir, os.path.basename(checkpoint_dir))
    
    # Merge and save the model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer successfully exported to {output_dir}")

if __name__ == "__main__":
    main()
