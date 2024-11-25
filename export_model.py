from unsloth import FastLanguageModel
import os
import torch
from transformers import logging
import argparse

def setup_model(model_path="artnoage/metastral"):
    """Initialize the base model with LoRA configuration"""
    try:
        # First try loading from local path
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=8192,
            dtype="bfloat16",
            load_in_4bit=False,
            load_in_8bit=True
        )
    except Exception as e:
        print(f"Could not load from local path, trying HuggingFace Hub: {e}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="artnoage/metastral",
            max_seq_length=8192,
            dtype="bfloat16",
            load_in_4bit=False,
            load_in_8bit=True
        )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=3407,
        use_rslora=False
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
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Model weights not found at {checkpoint_path}")
    
    # Load the LoRA weights
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    
    # Merge LoRA weights with base model
    model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the merged model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model and tokenizer successfully exported to {args.output_dir}")

if __name__ == "__main__":
    main()
