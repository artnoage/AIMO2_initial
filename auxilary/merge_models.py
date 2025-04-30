import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_models(model1_path, model2_path, output_path):
    """
    Merges two models of the same architecture using simple weight averaging.

    Args:
        model1_path (str): Path to the first model.
        model2_path (str): Path to the second model.
        output_path (str): Path to save the merged model.
    """
    print(f"Loading model 1 from {model1_path}...")
    model1 = AutoModelForCausalLM.from_pretrained(model1_path, torch_dtype=torch.bfloat16).to('cpu')

    print(f"Loading model 2 from {model2_path}...")
    model2 = AutoModelForCausalLM.from_pretrained(model2_path, torch_dtype=torch.bfloat16).to('cpu')

    print("Merging weights...")
    merged_state_dict = {}
    for key in model1.state_dict().keys():
        if key in model2.state_dict():
            merged_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2
        else:
            print(f"Warning: Key '{key}' not found in both models. Skipping.")
            merged_state_dict[key] = model1.state_dict()[key] # Keep the weight from model1 if not in model2

    # Create a new model instance and load the merged state dictionary
    # We can use the config from one of the original models
    merged_model = AutoModelForCausalLM.from_config(model1.config)
    merged_model.load_state_dict(merged_state_dict)

    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    print("Merging complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two models using simple weight averaging.")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to the first model.")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to the second model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model.")

    args = parser.parse_args()

    merge_models(args.model1_path, args.model2_path, args.output_path)
