import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import argparse

def find_latest_model():
    """Find the latest timestamp directory in models folder"""
    models_dir = 'models'
    timestamp_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) 
                     and d[0].isdigit()]  # Timestamps start with digits
    
    if not timestamp_dirs:
        raise ValueError(f"No timestamp directories found in {models_dir}")
    
    # Sort by timestamp (assuming YYYYMMDD_HHMMSS format)
    latest_dir = sorted(timestamp_dirs)[-1]
    return os.path.join(models_dir, latest_dir)

def main():
    parser = argparse.ArgumentParser(description='Quantize exported model to INT8')
    parser.add_argument('--model_dir', type=str,
                      help='Directory containing the model to quantize (optional, uses latest if not specified)')
    args = parser.parse_args()

    # Setup logging
    logging.set_verbosity_info()
    
    # Get model directory
    model_dir = args.model_dir if args.model_dir else find_latest_model()
    print(f"Quantizing model from: {model_dir}")
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    
    # Quantize to INT8
    model = model.to(torch.int8)
    
    # Create output directory
    timestamp = os.path.basename(model_dir)
    output_dir = os.path.join('models', 'quantized', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the quantized model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Quantized model and tokenizer successfully saved to {output_dir}")

if __name__ == "__main__":
    main()
