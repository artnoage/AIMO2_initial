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
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Could not load from local path, trying HuggingFace Hub: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            "artnoage/metastral",
            torch_dtype=torch.float32,
            trust_remote_code=True
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
    
    # Find the latest checkpoint
    checkpoint_dir = args.checkpoint_dir
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
    
    # Load the base model as a PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_path, torch_dtype=torch.float32)
    
    # Merge LoRA weights with base model and convert to float32
    model = model.merge_and_unload()
    model = model.to(torch.float32)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the merged model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model and tokenizer successfully exported to {args.output_dir}")

if __name__ == "__main__":
    main()
