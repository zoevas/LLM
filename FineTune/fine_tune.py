"""
🚀 Phi-2 Medical Fine-Tuning for RTX 3060 Laptop
539 Q&A examples → Medical AI Assistant
FP16 + LoRA + Gradient Checkpointing
"""


from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import torch

# -------------------------
# 0. Check GPU
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())
if device=="cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------
# 1. Load dataset
# -------------------------
dataset = load_dataset("json", data_files="QandA.jsonl", split="train")
print(f"📊 Loaded {len(dataset)} medical Q&A pairs")

# -------------------------
# 2. Load model & tokenizer in FP16
# -------------------------
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model in FP16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,        # FP16 for GPU
    trust_remote_code=True
)
model = model.to(device)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# -------------------------
# 3. LoRA setup
# -------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# 4. Format dataset
# -------------------------
def format_prompt(example):
    return f"### Instruction: {example['question']}\n### Response: {example['answer']}<|endoftext|>"

dataset = dataset.map(lambda x: {"text": format_prompt(x)})

# -------------------------
# 5. Training arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./phi2-medical-finetuned",
    num_train_epochs=2,
    per_device_train_batch_size=1,    # small batch to fit GPU
    gradient_accumulation_steps=8,    # simulate batch size of 8
    learning_rate=2e-4,
    warmup_steps=100,
    logging_strategy="steps",
    logging_steps=50,
    save_steps=500,
    max_steps=1500,
    optim="adamw_torch",
    fp16=True,
    save_total_limit=2,
)

# -------------------------
# 6. SFT Trainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

print("🎯 Training started... (grab coffee ☕)")
trainer.train()
trainer.save_model("./phi2-medical-finetuned")
print("✅ Model saved!")

# -------------------------
# 7. Inference / testing
# -------------------------
print("\n🩺 Testing medical assistant...")

# Reload base model in FP16 for inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
checkpoint_dir = "./phi2-medical-finetuned"

model = PeftModel.from_pretrained(base_model, checkpoint_dir)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Optimized pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device=="cuda" else -1,
    max_new_tokens=50,             # shorter → faster
    pad_token_id=tokenizer.eos_token_id
)

# Test prompts
prompts = [
    "what is hepatitis A?",
    "Patient fever 38.5°C, cough 3 days. Diagnosis?"
]

for prompt in prompts:
    result = pipe(f"### Instruction: {prompt}\n### Response:")
    print(f"\nQ: {prompt}")
    print(f"A: {result[0]['generated_text']}")