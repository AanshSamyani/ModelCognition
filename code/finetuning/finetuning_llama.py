import os
import json
import torch

from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", 
    dtype="auto", 
    load_in_8bit=True, 
    token=""
)

model = FastLanguageModel.get_peft_model(
    model, 
    r=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias="none", 
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora=False, 
    loftq_config=None
)

tokenizer = get_chat_template(
    tokenizer, 
    template_name = "llama-3.2"
)

dataset_path = "/nlsasfs/home/isea/isea10/aansh/introspection/data/finetuning/llama_1b_multi_token/training_dataset.jsonl"

with open(dataset_path, 'r') as f:
    data = json.load(f)
    

    






