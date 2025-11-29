import os
import json
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset

# -----------------------------
# 1️⃣ Load model in 8-bit + float16
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/nlsasfs/home/isea/isea10/aansh/deception_detection/weights/unsloth/Llama-3.2-1B-Instruct",
    dtype=None,          # <- force float16 to avoid bfloat16 issues
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit = True,
)

# -----------------------------
# 2️⃣ Apply LoRA PEFT
# -----------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# -----------------------------
# 3️⃣ Load tokenizer with chat template
# -----------------------------
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.2"
)

# -----------------------------
# 4️⃣ Load dataset
# -----------------------------
def load_multi_token_completion_dataset(dataset_path: str) -> Dataset:
    with open(dataset_path, "r") as f:
        data = json.load(f)     # list of dicts
    return Dataset.from_list(data)

dataset = load_multi_token_completion_dataset(
    "/nlsasfs/home/isea/isea10/aansh/introspection/data/finetuning/llama_1b_multi_token/training_dataset.jsonl"
)

# -----------------------------
# 5️⃣ Format dataset for Unsloth
# -----------------------------
def formatting_prompts_func(examples):
    messages = examples["messages"]
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
    return {"text": texts}

formatted_dataset = dataset.map(
    formatting_prompts_func,
    batched=True
)

# -----------------------------
# 6️⃣ Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    packing=False,
    jit=False,  # <- disable Unsloth JIT to avoid bnb matmul errors
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_strategy="epoch",
    ),
)

# -----------------------------
# 7️⃣ Train
# -----------------------------
trainer_stats = trainer.train()

# -----------------------------
# 8️⃣ Save
# -----------------------------
model.save_pretrained("/nlsasfs/home/isea/isea10/aansh/introspection_weights/finetuned_llama_1b_8bit")
tokenizer.save_pretrained("/nlsasfs/home/isea/isea10/aansh/introspection_weights/finetuned_llama_1b_8bit")
