import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer
#from transformers import logging
import re


model_type = "light"
model_name= "/Home/stat/laschos/AIMO2_initial/models/light/20250206_083807"
dataset_name="/Home/stat/laschos/AIMO2_initial/local_datasets/light/20250206_202750"


# Check if model_type is in paths
if model_type not in model_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in model_name path!")
    print("!"*80 + "\n")

if model_type not in dataset_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in dataset_name path!")
    print("!"*80 + "\n")

def _strip_prefix(s, pattern):
    # Use re.escape to escape any special characters in the pattern
    return re.sub(f"^{re.escape(pattern)}", "", s)

def main():
    # Set training type
    #logging.set_verbosity_info()
    
    def extract_xml_answer(solution: str) -> str:
        def find_matching_brace(s: str, start: int) -> int:
            """
            Find the index of the matching closing brace for an opening brace at the given start position.
            
            Args:
                s (str): The string to search.
                start (int): The index of the opening brace '{'.
            
            Returns:
                int: The index of the matching closing brace '}', or -1 if not found.
            """
            count = 1  # Initialize brace count
            i = start + 1  # Start searching after the opening brace
            while i < len(s) and count > 0:
                if s[i] == '{':
                    count += 1
                elif s[i] == '}':
                    count -= 1
                i += 1
            return i - 1 if count == 0 else -1

        # Pattern to find all occurrences of \boxed{ with proper escaping
        pattern = re.compile(r'\\boxed\{')
        for match in pattern.finditer(solution):
            start = match.end() - 1  # Position of the opening brace '{'
            end = find_matching_brace(solution, start)
            if end != -1:
                # Extract content between the braces
                content = solution[start + 1:end].strip()
                return content  # Return the first found boxed content

        return str(0)  # Return None if no boxed content is found

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        """Reward function that checks if the answer matches exactly"""
        # Completions are already strings, no need to access with indices
        extracted_responses = [extract_xml_answer(completion) for completion in completions]
        # Ensure answer is repeated for each completion if needed
        answers = [answer[i//trainer.num_generations] for i in range(len(completions))]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answers)]

    def length_penalty_func(prompts, completions, **kwargs) -> list[float]:
        """Penalty for very long solutions"""
        return [-0.001 * len(c) for c in completions]  # Small penalty per character
    
    def step_count_reward_func(prompts, completions, **kwargs) -> list[float]:
        """Reward solutions that show clear steps"""
        step_counts = [len(re.findall(r'Step \d+:', c)) for c in completions]
        return [0.1 * count for count in step_counts]  # 0.1 points per step


    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=16384,
        fast_inference = True,
        load_in_4bit=False,
        max_lora_rank = 128)
   
     # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,  # True or "unsloth" for very long context
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
    dataset = load_from_disk(dataset_name)
    def formatting_func(example):
        # Only keep the required fields
        required_fields = ['prompt', 'chosen', 'rejected', 'score_chosen', 'score_rejected','answer']
        filtered_example = {k: example[k] for k in required_fields if k in example}
        
        # Apply formatting
        filtered_example["prompt"] = tokenizer.apply_chat_template([filtered_example["prompt"]], tokenize=False)
        filtered_example["chosen"] = tokenizer.apply_chat_template([filtered_example["chosen"]], tokenize=False)
        filtered_example["rejected"] = tokenizer.apply_chat_template([filtered_example["rejected"]], tokenize=False)
        filtered_example["chosen"] = _strip_prefix(filtered_example["chosen"], "<s>")
        filtered_example["rejected"] = _strip_prefix(filtered_example["rejected"], "<s>")
        filtered_example['answer']= example['correct_answer']
        # Ensure all required fields are present
        missing_fields = [f for f in required_fields if f not in filtered_example]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        return filtered_example
    # Load and format dataset
    formatted_dataset = dataset.map(
        formatting_func,
        desc="Applying chat template"
    )
    
    # Concatenate original and shuffled datasets
    formatted_dataset = formatted_dataset.shuffle(seed=42)

    # Create timestamped output directory with model_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{model_type}/{timestamp}"


    # ORPO specific training arguments
    training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = output_dir,
)

    # Initialize ORPO trainer with multiple reward functions
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,  # Main correctness check
            length_penalty_func,      # Penalize verbosity
            step_count_reward_func    # Reward step-by-step solutions
        ],
        args=training_args,
        train_dataset=formatted_dataset,
    )
    # Train the model
    trainer.train()

    # Save both merged model and LoRA weights
    models_dir = "models"
    
    os.makedirs(os.path.join(models_dir, model_type), exist_ok=True)
    
    
    model_output_dir = os.path.join(models_dir, model_type, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {model_output_dir}")
    

if __name__ == "__main__":
    main()
