import os
import json
import torch

from tqdm import tqdm
from utils import find_assistant_token_end
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_single_token_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_path: str, 
    output_dir: str,
    output_filename: str,
) -> None:

    ## Dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    ## Get Log-Probs
    for prompt in tqdm(data):
        message = [
            {"role": "user", "content": prompt["prompt"]},
            {"role": "assistant", "content": prompt["completion_1"]},
        ]
        
        inputs_temp = tokenizer.apply_chat_template(message, tokenize=False)
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
        
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    
    
    