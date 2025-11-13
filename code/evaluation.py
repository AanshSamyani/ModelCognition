import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

## Dataset
dataset_path = "/nlsasfs/home/isea/isea10/aansh/introspection/data/exp1/single_token_completion_prompts_with_log_probs_filtered_only_single.json"

with open(dataset_path, 'r') as f:
    data = json.load(f)
    
## Model and Tokenizer
model_path = "/nlsasfs/home/isea/isea10/aansh/deception_detection/weights/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto", dtype = "auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

## Constructing ICL template
k = 6
for i, item in enumerate(data):
    