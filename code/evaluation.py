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
system_message = {
    "role": "system",
    "content": """
    You are a helpful assistant performing a classification task. 
    In each iteration, I will give you a short context and two candidate completions. I want you to choose the completion that your own brain (i.e., your internal thought model) thinks is more likely to come next in that context.
    Only output the completion with higher probability after each example.
    Do not explain your reasoning.
    """
}

k = 6
input_message = []
input_message.append(system_message)
for i, item in enumerate(data):
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
    
outputs_dict = {} 
output_path = "/nlsasfs/home/isea/isea10/aansh/introspection/results/single_only.json"

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
    
    outputs_dict[f"predicted_{i}"] = response
    outputs_dict[f"ground_truth_{i}"] = item["completion_1"] if item["logit_1"] > item["logit_2"] else item["completion_2"]
    
with open(output_path, 'w') as f:
    json.dump(outputs_dict, f, indent = 4)