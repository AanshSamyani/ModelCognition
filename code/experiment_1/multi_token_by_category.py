import os
import json
import torch

from tqdm import tqdm
from dataclasses import dataclass
from utils import find_assistant_token_end
from get_log_probs import get_single_token_log_probs, get_multi_token_log_probs
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation import evaluate_model_on_multi_token_completion, evaluate_model_on_single_token_completion, compute_metrics

@dataclass
class ExperimentConfig:
    model_name: str = "/nlsasfs/home/isea/isea10/aansh/deception_detection/weights/Llama-3.2-1B-Instruct"
    dataset_path: str = "/nlsasfs/home/isea/isea10/aansh/introspection/data/exp1/multi_token_completions"
    output_dir: str = "/nlsasfs/home/isea/isea10/aansh/introspection/results/exp_1_llama_1b/multi_token"
    log_probs_output_filename: str = "log_probs.json"
    evaluation_output_filename: str = "predictions.json"
    metrics_output_filename: str = "metrics.json"
    icl_examples: int = 10 
    is_user_prompt: bool = True
    
if __name__ == "__main__":    
    config = ExperimentConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    per_category_files = {}
    for i in range(1, 21):
        for file in os.listdir(config.dataset_path):
            file_id = int(file.split("_")[-1].split(".")[0])
            if file_id == i:
                per_category_files[i] = os.path.join(config.dataset_path, file)
                
    for category, dataset_path in per_category_files.items():
        category_output_dir = os.path.join(config.output_dir, f"category_{category}")
        os.makedirs(category_output_dir, exist_ok=True)
        
        get_multi_token_log_probs(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            output_dir=category_output_dir,
            output_filename=config.log_probs_output_filename,
            is_user_prompt=config.is_user_prompt,
        )
    
        evaluate_model_on_multi_token_completion(
            model=model,
            tokenizer=tokenizer,
            dataset_path=os.path.join(category_output_dir, config.log_probs_output_filename),
            output_dir=category_output_dir,
            output_filename=config.evaluation_output_filename,
            icl_examples=config.icl_examples,
        )
        
        compute_metrics(
            predictions_path=os.path.join(category_output_dir, config.evaluation_output_filename),
            output_dir = category_output_dir,
            output_filename = config.metrics_output_filename
        )
        
