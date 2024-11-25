from unsloth import FastLanguageModel
import os
import torch
from transformers import logging
import argparse

def setup_model():
    """Initialize the base model with LoRA configuration"""
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
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                      help='Directory containing the training checkpoints')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save the exported model')
    args = parser.parse_args()

    # Setup logging
    logging.set_verbosity_info()
    
    # Initialize model and tokenizer
    model, tokenizer = setup_model()
    
    # Load the trained LoRA weights
    checkpoint_path = os.path.join(args.checkpoint_dir, 'pytorch_model.bin')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    
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
