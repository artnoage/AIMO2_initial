from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

def main():
    # Load model and tokenizer
    model_name = "artnoage/metastral"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setup Mistral chat template
    tokenizer.chat_template = "<s>[INST] {{ messages[0]['content'] }} [/INST]\n{{ messages[1]['content'] }}</s>"

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False) 
                for convo in convos]
        return {"text": texts}

    # Load and format dataset
    print("Loading dataset...")
    dataset = load_dataset("Metaskepsis/sft", split="train")
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Analyze lengths
    print("\nAnalyzing sequence lengths...")
    lengths = []
    
    for example in tqdm(formatted_dataset):
        tokens = tokenizer(example["text"], return_length=True)
        lengths.append(tokens["length"])
    
    lengths = np.array(lengths)
    
    # Print statistics
    print("\nSequence Length Statistics:")
    print(f"Mean length: {lengths.mean():.1f}")
    print(f"Median length: {np.median(lengths):.1f}")
    print(f"Min length: {lengths.min()}")
    print(f"Max length: {lengths.max()}")
    print(f"95th percentile: {np.percentile(lengths, 95):.1f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.1f}")
    
    # Print histogram-like distribution
    print("\nLength Distribution:")
    percentiles = [50, 75, 90, 95, 99, 100]
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(lengths, p):.1f}")

if __name__ == "__main__":
    main()
