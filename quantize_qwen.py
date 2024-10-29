from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from transformers import AutoTokenizer
from datasets import load_dataset

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load model and tokenizer
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto", 
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load and preprocess calibration data
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"], 
        padding=False, 
        max_length=MAX_SEQUENCE_LENGTH, 
        truncation=True, 
        add_special_tokens=False
    )
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure and apply quantization
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the compressed model
SAVE_DIR = "qwen-math-7b-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Quantized model saved to: {SAVE_DIR}")
