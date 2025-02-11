import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth.chat_templates import get_chat_template
import sys
import logging
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict, Any, Optional, Tuple
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer, validate_solution


@dataclass
class GroupValidationStats:
    """Tracks validation statistics during training"""
    def __init__(self):
        self.total_batches = 0
        self.total_rewards = 0
        self.reward_distribution = {}
        self.similarity_stats = {
            'avg_similarity': 0.0,
            'unique_solutions': 0,
            'similar_solutions': 0
        }
        self.majority_stats = {
            'majority_agreement': 0,
            'split_decisions': 0,
            'no_consensus': 0
        }
        self.correctness_stats = {
            'correct_answers': 0,
            'incorrect_answers': 0,
            'parse_errors': 0
        }
        self.start_time = datetime.now()
    
    def update(self, rewards: List[float], similarities: Optional[torch.Tensor] = None, 
               correct_counts: Optional[Dict] = None):
        self.total_batches += 1
        for r in rewards:
            self.total_rewards += r
            r_rounded = round(r, 6)
            self.reward_distribution[r_rounded] = self.reward_distribution.get(r_rounded, 0) + 1
            
        if similarities is not None:
            avg_sim = similarities.mean().item()
            self.similarity_stats['avg_similarity'] = (
                (self.similarity_stats['avg_similarity'] * (self.total_batches - 1) + avg_sim) 
                / self.total_batches
            )
            unique = (similarities < 0.7).sum().item()
            similar = (similarities > 0.9).sum().item()
            self.similarity_stats['unique_solutions'] += unique
            self.similarity_stats['similar_solutions'] += similar
            
        if correct_counts is not None:
            if correct_counts['majority'] > correct_counts['minority']:
                self.majority_stats['majority_agreement'] += 1
            elif correct_counts['majority'] == correct_counts['minority']:
                self.majority_stats['split_decisions'] += 1
            else:
                self.majority_stats['no_consensus'] += 1
                
            self.correctness_stats['correct_answers'] += correct_counts['correct']
            self.correctness_stats['incorrect_answers'] += correct_counts['incorrect']
            self.correctness_stats['parse_errors'] += correct_counts['errors']
    
    def get_summary(self) -> str:
        elapsed = datetime.now() - self.start_time
        total_samples = sum(self.reward_distribution.values())
        if total_samples == 0:
            return "No samples processed yet"
            
        sorted_rewards = sorted(self.reward_distribution.items())
        reward_dist_str = "\n".join(
            f"  {reward:.6f}: {count} samples" 
            for reward, count in sorted_rewards
        )
        
        return (
            f"Training time: {elapsed}\n"
            f"Processed {self.total_batches} batches\n"
            f"Average reward: {self.total_rewards/total_samples:.6f}\n"
            f"\nReward Distribution:\n{reward_dist_str}\n"
            f"\nSimilarity Statistics:\n"
            f"  Average similarity: {self.similarity_stats['avg_similarity']:.4f}\n"
            f"  Unique solutions: {self.similarity_stats['unique_solutions']}\n"
            f"  Similar solutions: {self.similarity_stats['similar_solutions']}\n"
            f"\nMajority Statistics:\n"
            f"  Majority agreement: {self.majority_stats['majority_agreement']}\n"
            f"  Split decisions: {self.majority_stats['split_decisions']}\n"
            f"  No consensus: {self.majority_stats['no_consensus']}\n"
            f"\nCorrectness Statistics:\n"
            f"  Correct answers: {self.correctness_stats['correct_answers']}\n"
            f"  Incorrect answers: {self.correctness_stats['incorrect_answers']}\n"
            f"  Parse errors: {self.correctness_stats['parse_errors']}"
        )


class SolutionSimilarityChecker:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Freeze the embedding model's parameters to ensure they do not track gradients.
        for param in self.model.parameters():
            param.requires_grad = False

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512,  # Explicit max length
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    def compute_similarity_matrix(self, solutions: List[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.get_embeddings(solutions)
            return torch.mm(embeddings, embeddings.t()).detach()


def setup_training_logger(model_type: str) -> logging.Logger:
    """Setup logging configuration for training"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        f"{log_dir}/training_{timestamp}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def process_group_completions(completions: List[str], correct_answer: str) -> Tuple[List[bool], Dict]:
    """Process a group of completions and return correctness and statistics"""
    results = []
    correct_count = 0
    error_count = 0
    
    for completion in completions:
        try:
            model_answer = extract_answer_from_solution(completion)
            if model_answer is None:
                results.append(False)
                error_count += 1
                continue
                
            model_numeric, _ = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(correct_answer)
            
            if model_numeric is None or correct_numeric is None:
                results.append(False)
                error_count += 1
                continue
                
            is_correct = abs(model_numeric - correct_numeric) <= 1e-6
            results.append(is_correct)
            if is_correct:
                correct_count += 1
                
        except Exception:
            results.append(False)
            error_count += 1
            
    incorrect_count = len(completions) - correct_count - error_count
    majority = max(correct_count, incorrect_count)
    minority = min(correct_count, incorrect_count)
    
    stats = {
        'correct': correct_count,
        'incorrect': incorrect_count,
        'errors': error_count,
        'majority': majority,
        'minority': minority
    }
    
    return results, stats


def main():
    # Initialize similarity checker and statistics
    similarity_checker = SolutionSimilarityChecker()
    stats = GroupValidationStats()
    logger = setup_training_logger("group_grpo")
    
    # Setup callback for logging training statistics
    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logger.info(f"\nValidation Statistics:\n{stats.get_summary()}")
    
    class RewardFunction:
        def __init__(self, similarity_checker, stats):
            self.similarity_checker = similarity_checker
            self.stats = stats
            self.__name__ = "group_reward_function"  # Add name attribute
            
        def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
            # Get correct answers from kwargs
            correct_answers = kwargs.get('correct_answer', [''] * len(completions))
            
            # Group completions by prompt with indexed entries
            prompt_groups = {}
            for i, (comp, prom) in enumerate(zip(completions, prompts)):
                if prom not in prompt_groups:
                    prompt_groups[prom] = {'entries': [], 'answer': correct_answers[i]}
                prompt_groups[prom]['entries'].append({'completion': comp, 'index': i})
            
            # Process each group separately
            all_rewards = [0.0] * len(completions)
            
            for group in prompt_groups.values():
                # Extract completions while preserving indices
                group_completions = [entry['completion'] for entry in group['entries']]
                group_indices = [entry['index'] for entry in group['entries']]
                
                # Use the answer from kwargs since we're not getting it directly anymore
                correct_answer = kwargs.get('correct_answer', [''])[0]
                
                # Get correctness for each completion
                correctness_results, correct_stats = process_group_completions(
                    group_completions, correct_answer
                )
                
                # Compute similarity matrix for group
                similarity_matrix = self.similarity_checker.compute_similarity_matrix(group_completions)
                
                # Calculate rewards for each completion in group
                base_reward = 1.0
                diversity_bonus = 0.3
                majority_bonus = 0.2
                
                for i, (is_correct, idx) in enumerate(zip(correctness_results, group_indices)):
                    reward = 0.0
                    
                    if is_correct:
                        reward = base_reward
                        
                        # Add diversity bonus for unique correct solutions
                        similarities = similarity_matrix[i]
                        similarities[i] = 0  # Remove self-similarity
                        avg_similarity = similarities.mean().item()
                        
                        if avg_similarity < 0.7:  # Unique solution
                            reward += diversity_bonus
                        elif avg_similarity > 0.9:  # Very similar to others
                            reward -= diversity_bonus / 2
                            
                        # Add majority bonus if agrees with majority
                        if correct_stats['correct'] > len(group_completions) / 2:
                            reward += majority_bonus
                    
                    all_rewards[idx] = reward
                
                # Update statistics
                self.stats.update(
                    [all_rewards[i] for i in group['indices']], 
                    similarity_matrix,
                    correct_stats
                )
            
            return all_rewards
    
    # Load and format the dataset
    dataset = load_dataset("Metaskepsis/Numina_medium")
    
    def formatting_func(example):
        required_fields = ['prompt', 'answer']
        filtered_example = {k: example[k] for k in required_fields if k in example}
        
        solver_prompt = (
            "Here is a mathematical problem:\n\n"
            f"{example['problem']}\n\n"
            "Could you help me solve this from start to finish? First, let's analyze the problem, "
            "then walk through the solution step-by-step using LaTeX notation. "
            "Don't forget to put the final answer in a box using \\boxed{}"
        )
        filtered_example["prompt"] = f"[INST]{solver_prompt}[/INST]"
        filtered_example['answer'] = example['answer']
        return filtered_example
    
    formatted_dataset = dataset['train'].map(
        formatting_func,
        desc="Applying chat template"
    )
    
    # Load the model and tokenizer using FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/Home/stat/laschos/AIMO2_initial/models/light/20250206_212611",
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_lora_rank=64
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    
    # Setup chat template for the tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True
    )
    
    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/group_grpo/{timestamp}"
    
    # Training arguments
    training_args = GRPOConfig(
        use_vllm=True,
        torch_empty_cache_steps=10,
        learning_rate=3e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=3,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=2048,
        max_completion_length=2048,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=output_dir,
    )
    
    # Initialize trainer with the reward function
    reward_func = RewardFunction(similarity_checker, stats)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback()]
    )
    
    # Train the model
    trainer.train()
    
    # Save the merged model
    models_dir = "models"
    os.makedirs(os.path.join(models_dir, "group_grpo"), exist_ok=True)
    model_output_dir = os.path.join(models_dir, "group_grpo", timestamp)
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    logger.info(f"Merged model saved to {model_output_dir}")

    
if __name__ == "__main__":
    main()
