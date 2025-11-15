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
        
def get_multi_token_log_probs(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    dataset_path: str, 
    output_dir: str,
    output_filename: str,
    is_user_prompt: bool = True,
) -> None:
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
        
    for prompt in tqdm(data):
        if is_user_prompt:
            user_prompt = f"""
            Complete the following sentence:
            {prompt["prompt"]} 
            """
            message_1 = [
                {"role": "user", "content": user_prompt}, 
                {"role": "assistant", "content": prompt["completion_1"]},
            ]
            
            message_2 = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": prompt["completion_2"]}
            ]
        
        else:
            user_prompt = f"""
            Write down a statement.
            """
            
            message_1 = [
                {"role": "user", "content": user_prompt}, 
                {"role": "assistant", "content": prompt["completion_1"]},
            ]
            
            message_2 = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": prompt["completion_2"]}
            ]
            
        inputs_temp_1 = tokenizer.apply_chat_template(message_1, tokenize=False)
        inputs_1 = tokenizer(inputs_temp_1, return_tensors="pt").to(model.device)
        
        start_token_idx_1 = find_assistant_token_end(inputs_1["input_ids"][0].tolist())
        end_token_idx_1 = len(inputs_1["input_ids"][0]) - 2
        
        outputs_1 = model(**inputs_1)
        logits_1 = outputs_1.logits[0]
        
        log_probs_1 = 1
        for token_idx in range(start_token_idx_1, end_token_idx_1): 
            log_probs_1 = log_probs_1 * torch.log_softmax(logits_1[token_idx], dim=-1)[inputs_1["input_ids"][0][token_idx + 1]].item()
            
        inputs_temp_2 = tokenizer.apply_chat_template(message_2, tokenize=False)
        inputs_2 = tokenizer(inputs_temp_2, return_tensors="pt".to(model.device))
        
        start_token_idx_2 = find_assistant_token_end(inputs_2["input_ids"][0].tolist())
        end_token_idx_2 = len(inputs_2["input_ids"][0]) - 2
        
        outputs_2 = model(**inputs_2)
        logits_2 = outputs_2.logits[0]
        
        log_probs_2 = 1
        for token_idx in range(start_token_idx_2, end_token_idx_2): 
            log_probs_2 = log_probs_2 * torch.log_softmax(logits_2[token_idx], dim=-1)[inputs_2["input_ids"][0][token_idx + 1]].item()
            
        prompt["log_probs_1"] = log_probs_1
        prompt["log_probs_2"] = log_probs_2           
        
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4) 
            


    