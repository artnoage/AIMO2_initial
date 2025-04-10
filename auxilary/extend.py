# Ensure you have transformers and torch installed:
# pip install transformers torch

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import os
import json

# --- Configuration ---
model_directory = "/Home/stat/laschos/math/AIMO2_initial/models/O1" # Replace with the path to your model files
output_directory = "/Home/stat/laschos/math/AIMO2_initial/models/O2"      # Directory to save the new 30-layer model
new_total_layers = 34
reinit_std_dev = 0.01 # Standard deviation for re-initializing the new layers

# --- Ensure config.json and index file exist (rename if needed) ---
config_path = os.path.join(model_directory, "config.json")
if not os.path.exists(config_path):
    txt_config_path = os.path.join(model_directory, "config.txt")
    if os.path.exists(txt_config_path):
        print(f"Renaming {txt_config_path} to {config_path}")
        os.rename(txt_config_path, config_path)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path} or {txt_config_path}")

index_path = os.path.join(model_directory, "model.safetensors.index.json")
if not os.path.exists(index_path):
    txt_index_path = os.path.join(model_directory, "model.safetensors.index.txt")
    if os.path.exists(txt_index_path):
        print(f"Renaming {txt_index_path} to {index_path}")
        os.rename(txt_index_path, index_path)
    else:
        # Note: index might not be strictly necessary for loading if all weights
        # are in the directory, but renaming is good practice.
        print(f"Warning: Model index file not found at {index_path} or {txt_index_path}")


# --- Load original config and modify it ---
print(f"Loading configuration from: {model_directory}")
config = AutoConfig.from_pretrained(model_directory)

original_num_layers = config.num_hidden_layers
print(f"Original number of layers: {original_num_layers}")

if new_total_layers <= original_num_layers:
     raise ValueError(f"New total layers ({new_total_layers}) must be greater than original ({original_num_layers})")

# Modify the configuration object in memory
config.num_hidden_layers = new_total_layers
print(f"Modified config to have {config.num_hidden_layers} layers.")

# --- Load model using modified config and original weights ---
# This will load weights for layers 0-(original_num_layers-1)
# and randomly initialize layers original_num_layers to (new_total_layers-1)
print(f"Loading model from {model_directory} using the modified {new_total_layers}-layer config...")
print("Expect warnings about weights not being initialized for new layers.")

model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    config=config,
    ignore_mismatched_sizes=False, # Keep False to ensure core dimensions match
    torch_dtype=torch.bfloat16 # Adjust dtype if necessary
    # device_map='auto' # Optional: for memory management
)

print("Model loaded. Layers 0 to", original_num_layers - 1, "have pre-trained weights.")
print("Layers", original_num_layers, "to", new_total_layers - 1, "have default random initialization.")

# --- Explicitly re-initialize the weights of the new layers ---
print(f"\nExplicitly re-initializing layers {original_num_layers} to {new_total_layers - 1} with std_dev={reinit_std_dev}...")

# Ensure the model structure actually has the layers (it should)
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    actual_layers_in_model = len(model.model.layers)
    if actual_layers_in_model != new_total_layers:
         print(f"Warning: Model structure has {actual_layers_in_model} layers, expected {new_total_layers}.")
         # Adjust loop range if necessary, though it should match config
         layers_to_reinit_indices = range(original_num_layers, min(new_total_layers, actual_layers_in_model))
    else:
         layers_to_reinit_indices = range(original_num_layers, new_total_layers)

    for layer_idx in layers_to_reinit_indices:
        try:
            layer_to_reinit = model.model.layers[layer_idx]
            num_params_reinit = 0
            for name, param in layer_to_reinit.named_parameters():
                if param.requires_grad: # Only re-initialize trainable parameters
                    if 'weight' in name:
                        torch.nn.init.normal_(param.data, mean=0.0, std=reinit_std_dev)
                        num_params_reinit += param.numel()
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param.data)
                        num_params_reinit += param.numel()
            print(f"  - Re-initialized layer {layer_idx} ({num_params_reinit} parameters)")
        except IndexError:
            print(f"Error: Could not access layer index {layer_idx}. Check model structure.")
        except Exception as e:
            print(f"Error re-initializing layer {layer_idx}: {e}")
else:
    print("Warning: Could not find 'model.model.layers' attribute. Check model architecture access.")

print("Manual re-initialization of new layers complete.")

# --- Save the new 30-layer model ---
print(f"\nSaving the modified {new_total_layers}-layer model to: {output_directory}")
# This saves the merged state: trained layers 0-27 + re-initialized layers 28-29
model.save_pretrained(output_directory, safe_serialization=True)

# Also save the modified config alongside the new model
config.save_pretrained(output_directory)

print("New model saved successfully.")
print(f"Output files (including new config.json and model.safetensors.index.json) are in: {output_directory}")