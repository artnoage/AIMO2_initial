import os
import sys
import argparse
from datetime import datetime
from unsloth import FastLanguageModel, is_bfloat16_supported

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description='Export trained model from checkpoint')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Base model name/path')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: models/<model_type>/<timestamp>)')
    args = parser.parse_args()

    # Extract model type from path
    model_type = "tutor" 

    # Load the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=8192,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth")

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    # Load checkpoint
    model.load_adapter(args.checkpoint, "default")

    # Setup output directory
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("models", model_type, timestamp)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Save the merged model
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {output_dir}")

if __name__ == "__main__":
    main()
