import os
import json
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model_on_single_token_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    output_dir: str,
    output_filename: str,
    icl_examples: int = 6,
):

    with open(dataset_path, 'r') as f:
        data = json.load(f)
        

    ## Constructing ICL template
    system_message = {
        "role": "system",
        "content": """
        You are a helpful assistant performing a classification task. 
        In each iteration, I will give you a short context and two candidate completions. I want you to choose the completion that your own brain (i.e., your internal thought model) thinks is more likely to come next in that context.
        Only output the completion with higher probability after each example.
        Do not explain your reasoning.
        """
    }

    k = icl_examples
    input_message = []
    input_message.append(system_message)
    for i, item in tqdm(enumerate(data)):
        if i > k:
            break
        
        user_message = {
            "role": "user",
            "content": f"Context: {item['prompt']}\nCompletion 1: {item['completion_1']}\nCompletion 2: {item['completion_2']}\nWhich completion is more likely according to your internal thought model? Only output the completion with higher probability."
        }
        input_message.append(user_message)
        
        assistant_message = {
            "role": "assistant",
            "content": item["completion_1"] if item["logit_1"] > item["logit_2"] else item["completion_2"]
        }
        input_message.append(assistant_message)
        
    outputs_dict = {"predicted": [], "ground_truth": []} 
    output_path = os.path.join(output_dir, output_filename)

    for i, item in enumerate(data[k:]):
        user_message = {
            "role": "user",
            "content": f"Context: {item['prompt']}\nCompletion 1: {item['completion_1']}\nCompletion 2: {item['completion_2']}\nWhich completion is more likely according to your internal thought model? Only output the completion with higher probability."
        }
        new_message = input_message + [user_message]
        
        inputs_temp = tokenizer.apply_chat_template(new_message, tokenize = False)
        inputs = tokenizer(inputs_temp, return_tensors = "pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens = 50)
        response = tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens = True).strip()
        
        outputs_dict["predicted"].append(response)
        outputs_dict["ground_truth"].append(item["completion_1"] if item["logit_1"] > item["logit_2"] else item["completion_2"])
        
    with open(output_path, 'w') as f:
        json.dump(outputs_dict, f, indent = 4)
        
