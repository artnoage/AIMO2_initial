import os
import wandb
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import sys
import os
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer, validate_solution
#from transformers import logging
import re


model_type = "light"
model_name= "/Home/stat/laschos/AIMO2_initial/models/light/20250206_212611"
dataset_name="Metaskepsis/Numina_medium"


# Check if model_type is in paths
if model_type not in model_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in model_name path!")
    print("!"*80 + "\n")

if model_type not in dataset_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in dataset_name path!")
    print("!"*80 + "\n")

def main():
    # Set training type
    #logging.set_verbosity_info()
    
    def extract_xml_answer(solution: str) -> str:
        try:
            # First extract the boxed answer
            boxed_answer = extract_answer_from_solution(solution)
            if boxed_answer is None:
                return -123456789101110
                
            # Then convert to numeric value
            numeric_value, _ = extract_numeric_answer(boxed_answer)
            if numeric_value is None:
                return -123456789101111
                
            return numeric_value
        except Exception:
            return -123456789101112

    def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
        """Reward function that checks if the answer matches exactly"""
        # Completions are already strings, no need to access with indices
        extracted_responses = [extract_xml_answer(completion) for completion in completions]
        # Ensure answer is repeated for each completion if needed
        answers = [answer[i//trainer.num_generations] for i in range(len(completions))]
        # Convert answers to numeric values
        
        numeric_answers = [extract_numeric_answer(a)[0] if extract_numeric_answer(a)[0] is not None else -123456789101112 for a in answers]
        return [2.0 if abs(r - a) < 1e-6 else 0.0 for r, a in zip(extracted_responses, numeric_answers)]

    def length_penalty_func(completions, **kwargs) -> list[float]:
        """Penalty for very long solutions"""
        return [-0.0001 * len(c) for c in completions]  # Small penalty per character
    
    def validation_reward_func(completions, **kwargs) -> list[float]:
        """Reward solutions that pass validation checks"""
        return [0.2 if validate_solution(completion)[0] else 0.0 for completion in completions]


    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=6496,
        fast_inference = True,
        load_in_4bit=False,
        use_gradient_checkpointing= "unsloth",
        max_lora_rank = 64 )
   
     # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None)
        

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True)
    
    # Load dataset - adjust path as needed
    #dataset = load_dataset("Metaskepsis/orpo", split="train")
    dataset = load_dataset(dataset_name)
    def formatting_func(example):
        # Only keep the required fields
        required_fields = ['prompt', 'answer']
        filtered_example = {k: example[k] for k in required_fields if k in example}
        
        # First apply the solver prompt, then wrap in INST tags
        solver_prompt = (
            "Here is a mathematical problem:\n\n"
            f"{example['problem']}\n\n"
            "Could you help me solve this from start to finish? First, let's analyze the problem, "
            "then walk through the solution step-by-step using LaTeX notation. "
            "Don't forget to put the final answer in a box using \\boxed{}"
        )
        filtered_example["prompt"] = f"[INST]{solver_prompt}[/INST]"
        filtered_example['answer'] = example['answer']
        # Ensure all required fields are present
        missing_fields = [f for f in required_fields if f not in filtered_example]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        return filtered_example
    # Load and format dataset
    formatted_dataset = dataset['train'].map(
        formatting_func,
        desc="Applying chat template"
    )
    
    # Print first entry tokenization
    first_entry = formatted_dataset[0]
    print("\nFirst entry tokenization:")
    print("Original:", first_entry['prompt'])
    tokenized = tokenizer(first_entry['prompt'])
    print("Tokenized:", tokenized)
    print("Decoded:", tokenizer.decode(tokenized['input_ids']))
    
    # Concatenate original and shuffled datasets
    formatted_dataset = formatted_dataset.shuffle(seed=42)

    # Create timestamped output directory with model_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{model_type}/{timestamp}"

    # Initialize wandb
    wandb.init(
        project="solution_grpo",
        name=f"solution_grpo_{timestamp}",
        config={
            "model_type": model_type,
            "dataset": dataset_name,
            "base_reward": 2.0,
            "validation_reward": 0.2,
            "length_penalty_factor": 0.0001
        }
    )

    # GRPO specific training arguments
    training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    torch_empty_cache_steps=10,
    learning_rate = 3e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.05,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 3,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations =5, # Decrease if out of memory
    max_prompt_length = 1348,
    max_completion_length = 5148,
    num_train_epochs = 1, # Set to 1 for a full training run
    save_steps = 250, 
    max_grad_norm = 0.1,
    report_to = "wandb", # Using Weights & Biases
    output_dir = output_dir,
)

    # Setup callback for logging training statistics
    class LoggingCallback(TrainerCallback):
        def __init__(self, save_frequency=100):
            self.save_frequency = save_frequency
            self.step = 0
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            self.step += 1
            
            # Log to wandb
            if logs:
                # Add reward function specific metrics
                if 'rewards/0' in logs:  # Correctness reward
                    wandb.log({'correctness_reward': logs['rewards/0']})
                if 'rewards/1' in logs:  # Length penalty
                    wandb.log({'length_penalty': logs['rewards/1']})
                if 'rewards/2' in logs:  # Validation reward
                    wandb.log({'validation_reward': logs['rewards/2']})
                
                wandb.log(logs)

    # Initialize GRPO trainer with multiple reward functions
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,  # Main correctness check
            length_penalty_func,      # Penalize verbosity
            validation_reward_func    # Reward valid solutions
        ],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback(save_frequency=100)]
    )
    # Train the model
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {str(e)}")
        wandb.finish()
        raise

    # Save both merged model and LoRA weights
    models_dir = "models"
    
    os.makedirs(os.path.join(models_dir, model_type), exist_ok=True)
    
    
    model_output_dir = os.path.join(models_dir, model_type, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {model_output_dir}")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
