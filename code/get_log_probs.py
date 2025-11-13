import os
import json
import torch

from utils import find_assistant_token_end
from transformers import AutoModelForCausalLM, AutoTokenizer

## Dataset
dataset_path = "/nlsasfs/home/isea/isea10/aansh/introspection/data/exp1/single_token_completion_prompts.json"

with open(dataset_path, "r") as f:
    data = json.load(f)
    
## Model and Tokenizer
model_path = "/nlsasfs/home/isea/isea10/aansh/deception_detection/weights/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

## Get Log-Probs
for prompt in data:
    message_1 = [
        {"role": "user", "content": prompt["prompt"]},
        {"role": "assistant", "content": prompt["completion_1"]},
    ]
    
    inputs_temp = tokenizer.apply_chat_template(message_1, tokenize=False)
    inputs = tokenizer(inputs_temp, return_tensors="pt").to(model.device)
    detect_token_idx = find_assistant_token_end(inputs["input_ids"][0].tolist())
    token_idx_1 = tokenizer.encode(prompt["completion_1"])[1]
    token_idx_2 = tokenizer.encode(prompt["completion_2"])[1]

    outputs = model(**inputs)
    logits_detect_idx_1 = outputs.logits[0, detect_token_idx, token_idx_1]
    
    logits_detect_idx_2 = outputs.logits[0, detect_token_idx, token_idx_2]
    
    prompt["token_1"] = tokenizer.decode(token_idx_1)
    prompt["token_2"] = tokenizer.decode(token_idx_2)
    
    prompt["logit_1"] = logits_detect_idx_1.item()
    prompt["logit_2"] = logits_detect_idx_2.item()
    
    prompt["log_prob_1"] = torch.log_softmax(outputs.logits[0, detect_token_idx], dim=-1)[token_idx_1].item()
    prompt["log_prob_2"] = torch.log_softmax(outputs.logits[0, detect_token_idx], dim=-1)[token_idx_2].item()
    
output_path = "/nlsasfs/home/isea/isea10/aansh/introspection/data/exp1/single_token_completion_prompts_with_log_probs.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
        
    
    
    