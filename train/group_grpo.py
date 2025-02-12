import os
import json
from pathlib import Path
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
    def __init__(self, output_dir: str):
        self.total_batches = 0
        self.start_time = datetime.now()
        self.output_dir = output_dir
    
    def update(self, rewards: List[float], similarities: Optional[torch.Tensor] = None, 
               correct_counts: Optional[Dict] = None, group_stats: Optional[Dict] = None):
        """Store statistics for current batch"""
        batch_stats = {
            'batch_id': self.total_batches,
            'timestamp': datetime.now().isoformat(),
            'rewards': rewards,
            'reward_distribution': {},
            'similarity_stats': {},
            'correctness_stats': {},
            'group_stats': group_stats  # Detailed statistics about the group
        }
        
        # Record reward distribution
        for r in rewards:
            r_rounded = round(r, 6)
            batch_stats['reward_distribution'][r_rounded] = batch_stats['reward_distribution'].get(r_rounded, 0) + 1
            
        # Record similarity stats if available
        if similarities is not None:
            batch_stats['similarity_stats'] = {
                'avg_similarity': similarities.mean().item(),
                'unique_solutions': (similarities < 0.7).sum().item(),
                'similar_solutions': (similarities > 0.9).sum().item()
            }
            
        # Record correctness stats if available
        if correct_counts is not None:
            batch_stats['correctness_stats'] = correct_counts.copy()
            
        # Save batch statistics immediately
        self.save_batch_statistics(batch_stats)
        self.total_batches += 1
    
    def save_batch_statistics(self, batch_stats: Dict):
        """Update the running statistics file with new batch stats"""
        import json
        from pathlib import Path
        
        # Create stats directory if it doesn't exist
        stats_dir = Path(self.output_dir) / "statistics"
        stats_dir.mkdir(exist_ok=True)
        stats_file = stats_dir / "training_stats.json"
        
        # Load existing stats or create new
        if stats_file.exists():
            with open(stats_file) as f:
                all_stats = json.load(f)
        else:
            all_stats = {
                'batches': [],
                'start_time': self.start_time.isoformat(),
                'last_update': None
            }
        
        # Add new batch stats
        all_stats['batches'].append(batch_stats)
        all_stats['last_update'] = datetime.now().isoformat()
        
        # Save updated stats
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

    def get_summary(self) -> str:
        """Get a summary of recent batches"""
        from pathlib import Path
        import json
        
        # Load stats file
        stats_dir = Path(self.output_dir) / "statistics"
        stats_file = stats_dir / "training_stats.json"
        if not stats_file.exists():
            return "No statistics available yet"
            
        with open(stats_file) as f:
            all_stats = json.load(f)
            
        # Get last 5 batches
        last_batches = all_stats['batches'][-5:]
        if not last_batches:
            return "No batches found"
            
        # Aggregate recent statistics
        recent_stats = {
            'rewards': [],
            'groups': set(),
            'correct_count': 0,
            'total_count': 0,
            'unique_solutions': 0,
            'similar_solutions': 0
        }
        
        for batch in last_batches:
            recent_stats['rewards'].extend(batch['rewards'])
            if batch.get('group_stats', {}).get('group_id'):
                recent_stats['groups'].add(batch['group_stats']['group_id'])
            if batch.get('correctness_stats'):
                recent_stats['correct_count'] += batch['correctness_stats'].get('correct', 0)
                recent_stats['total_count'] += sum(batch['correctness_stats'].values())
            if batch.get('similarity_stats'):
                recent_stats['unique_solutions'] += batch['similarity_stats'].get('unique_solutions', 0)
                recent_stats['similar_solutions'] += batch['similarity_stats'].get('similar_solutions', 0)
        
        avg_reward = sum(recent_stats['rewards']) / len(recent_stats['rewards']) if recent_stats['rewards'] else 0
        accuracy = (recent_stats['correct_count'] / recent_stats['total_count'] * 100) if recent_stats['total_count'] else 0
        
        return (
            f"Recent Statistics (last 5 batches):\n"
            f"Training time: {datetime.now() - self.start_time}\n"
            f"Total batches: {self.total_batches}\n"
            f"Unique groups: {len(recent_stats['groups'])}\n"
            f"Average reward: {avg_reward:.4f}\n"
            f"Accuracy: {accuracy:.1f}%\n"
            f"Unique solutions: {recent_stats['unique_solutions']}\n"
            f"Similar solutions: {recent_stats['similar_solutions']}\n"
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
            max_length=512,
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
    # Initialize similarity checker
    similarity_checker = SolutionSimilarityChecker()
    logger = setup_training_logger("group_grpo")
    
    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/group_grpo/{timestamp}"
    
    # Initialize statistics
    stats = GroupValidationStats(output_dir)
    
    # Setup callback for logging training statistics
    class LoggingCallback(TrainerCallback):
        def __init__(self, stats, output_dir, save_frequency=100):
            self.stats = stats
            self.output_dir = output_dir
            self.save_frequency = save_frequency
            self.step = 0
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Log to local file
            logger.info(f"\nValidation Statistics:\n{self.stats.get_summary()}")
            self.step += 1
            
            # Log to wandb
            if logs:
                wandb.log(logs)
                
            # Also log our custom statistics to wandb
            stats_dir = Path(self.output_dir) / "statistics"
            stats_file = stats_dir / "training_stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    all_stats = json.load(f)
                    if all_stats['batches']:
                        last_batch = all_stats['batches'][-1]
                        wandb.log({
                            "avg_reward": sum(last_batch['rewards']) / len(last_batch['rewards']),
                            "correct_answers": last_batch['correctness_stats'].get('correct', 0),
                            "incorrect_answers": last_batch['correctness_stats'].get('incorrect', 0),
                            "unique_solutions": last_batch['similarity_stats'].get('unique_solutions', 0),
                            "similar_solutions": last_batch['similarity_stats'].get('similar_solutions', 0)
                        })
                
        def on_train_end(self, args, state, control, **kwargs):
            # No need for final save since batch_statistics are saved immediately
            pass
    
    class RewardFunction:
        def __init__(self, similarity_checker, stats):
            self.similarity_checker = similarity_checker
            self.stats = stats
            self.__name__ = "group_reward_function"  # Add name attribute
            
        def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
            # Get correct answers from kwargs
            correct_answers = kwargs.get('correct_answer', [''] * len(completions))
            logger.info(f"Processing {len(correct_answers)} answers, first: {correct_answers[0]}, last: {correct_answers[-1]}")
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
                
                # Use the answer from the group since we already have it
                correct_answer = group['answer']
                # Get correctness for each completion
                correctness_results, correct_stats = process_group_completions(
                    group_completions, correct_answer
                )
                logger.info(f"\nGroup Analysis:")
                logger.info(f"Correct answer: {correct_answer}")
                logger.info(f"Correctness results: {correctness_results}")
                logger.info(f"Correct stats: {correct_stats}")
                
                # Log first few chars of each completion
                for i, comp in enumerate(group_completions):
                    answer = extract_answer_from_solution(comp)
                    logger.info(f"\nCompletion {i} ({'correct' if correctness_results[i] else 'incorrect'}):")
                    logger.info(f"Found answer: {answer}")
                    logger.info(f"First 100 chars: {comp[:100]}...")
                
                # Compute similarity matrix for group
                similarity_matrix = self.similarity_checker.compute_similarity_matrix(group_completions)
                logger.info("\nSimilarity matrix:")
                logger.info(f"{similarity_matrix}")
                
                # Calculate rewards for each completion in group
                base_reward = 3.0
                diversity_bonus = 0.3
                majority_bonus = 0.2
                
                # Calculate rewards for each completion
                for i, (is_correct, idx) in enumerate(zip(correctness_results, group_indices)):
                    reward = 0.0
                    
                    # Calculate majority agreement first
                    is_in_majority = (is_correct and correct_stats['correct'] > len(group_completions) / 2) or \
                                   (not is_correct and correct_stats['incorrect'] > len(group_completions) / 2)
                    
                    # Initialize reward components dictionary
                    reward_components = {
                        'base': base_reward if is_correct else 0.0,  # Base reward for correct answers
                        'majority_bonus': 0.0,
                        'diversity_bonus': 0.0,
                        'total': 0.0
                    }
                    
                    # Add majority bonus
                    if is_in_majority:
                        reward_components['majority_bonus'] = majority_bonus if is_correct else majority_bonus * 0.1
                    
                    # Add diversity bonus
                    similarities = similarity_matrix[i]
                    similarities[i] = 0  # Remove self-similarity
                    avg_similarity = similarities.mean().item()
                    
                    if avg_similarity < 0.7:  # Unique solution
                        reward_components['diversity_bonus'] = diversity_bonus if is_correct else diversity_bonus * 0.1
                    elif avg_similarity > 0.9:  # Very similar to others
                        reward_components['diversity_bonus'] = -diversity_bonus / 2 if is_correct else -diversity_bonus * 0.05
                    
                    # Calculate total reward
                    reward = sum(reward_components.values())
                    reward_components['total'] = reward
                    
                    all_rewards[idx] = reward
                    logger.info(f"\nReward calculation for completion {i}:")
                    logger.info(f"Is correct: {is_correct}")
                    logger.info(f"Is in majority: {is_in_majority}")
                    logger.info(f"Average similarity: {avg_similarity:.3f}")
                    logger.info("Reward components:")
                    for component, value in reward_components.items():
                        logger.info(f"  {component}: {value:.3f}")
                
                # Prepare detailed statistics
                group_stats = {
                    'group_id': str(hash(str(group['answer']) + str(group_completions[0][:100]))),
                    'correct_answer': correct_answer,
                    'completions': [{
                        'index': idx,
                        'is_correct': is_correct,
                        'is_in_majority': (is_correct and correct_stats['correct'] > len(group_completions) / 2) or \
                                        (not is_correct and correct_stats['incorrect'] > len(group_completions) / 2),
                        'avg_similarity': float(similarity_matrix[i].mean().item()),
                        'reward_components': reward_components,
                        'answer': extract_answer_from_solution(comp)
                    } for i, (comp, is_correct, idx) in enumerate(zip(group_completions, correctness_results, group_indices))],
                    'correctness_stats': correct_stats,
                    'similarity_matrix': similarity_matrix.tolist()
                }
                
                # Update statistics
                self.stats.update(
                    [all_rewards[idx] for entry in group['entries']], 
                    similarity_matrix,
                    correct_stats,
                    group_stats=group_stats
                )
            
            # Print rewards grouped by 8
            print("\nAll rewards (grouped by 8):")
            for i in range(0, len(all_rewards), 8):
                group = all_rewards[i:i+8]
                print(f"Group {i//8}: {[round(r, 3) for r in group]}")
            
            return all_rewards
    
    # Load and format the dataset
    dataset = load_dataset("Metaskepsis/Numina_very_hard")
    
    def formatting_func(example):
        required_fields = ['prompt', 'correct_answer']
        filtered_example = {k: example[k] for k in required_fields if k in example}
        
        solver_prompt = (
            "Here is a mathematical problem:\n\n"
            f"{example['problem']}\n\n"
            "Could you help me solve this from start to finish? First, let's analyze the problem, "
            "then walk through the solution step-by-step using LaTeX notation. "
            "Don't forget to put the final answer in a box using \\boxed{}"
        )
        filtered_example["prompt"] = f"[INST]{solver_prompt}[/INST]"
        filtered_example['correct_answer'] = example['answer']
        return filtered_example
    
    formatted_dataset = dataset['train'].map(
        formatting_func,
        desc="Applying chat template"
    )
    formatted_dataset = formatted_dataset.shuffle(seed=42)
    # Take first 3000 entries
    formatted_dataset = formatted_dataset.select(range(3000))
    # Load the model and tokenizer using FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/Home/stat/laschos/AIMO2_initial/models/light/20250209_172917",
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=8,
        max_prompt_length=2048,
        max_completion_length=2048,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
    )
    
    # Initialize wandb
    import wandb
    wandb.init(
        project="group_grpo",
        name=f"group_grpo_{timestamp}",
        config={
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_generations": training_args.num_generations,
            "base_reward": 3.0,
            "diversity_bonus": 0.3,
            "majority_bonus": 0.2
        }
    )
    
    # Initialize trainer with the reward function
    reward_func = RewardFunction(similarity_checker, stats)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback(stats=stats, output_dir=output_dir, save_frequency=2)]
    )
    
    # Train the model
    trainer.train()
    
    # Save the merged model
    models_dir = "models"
    os.makedirs(os.path.join(models_dir, "group_grpo"), exist_ok=True)
    model_output_dir = os.path.join(models_dir, "group_grpo", timestamp)
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    logger.info(f"Merged model saved to {model_output_dir}")
    
    # Close wandb run
    wandb.finish()

    
if __name__ == "__main__":
    main()
