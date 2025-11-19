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
    model_name: str = "/nlsasfs/home/isea/isea10/aansh/deception_detection/weights/Llama-3.3-70B-Instruct/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
    dataset_path: str = "/nlsasfs/home/isea/isea10/aansh/introspection/data/exp1/multi_token_completions/character_emotion_ambiguity_category_6.json"
    output_dir: str = "/nlsasfs/home/isea/isea10/aansh/introspection/results/exp_1_llama_70b/multi_token/category_6"
    log_probs_output_filename: str = "log_probs.json"
    evaluation_output_filename: str = "predictions.json"
    metrics_output_filename: str = "metrics.json"
    icl_examples: int = 10 
    is_single_token: bool = False
    is_user_prompt: bool = True
    
def run_experiment(config: ExperimentConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    if config.is_single_token: 
        get_single_token_log_probs(
            model=model,
            tokenizer=tokenizer,
            dataset_path=config.dataset_path,
            output_dir=config.output_dir,
            output_filename=config.log_probs_output_filename,
        )
        
        evaluate_model_on_single_token_completion(
        model=model,
        tokenizer=tokenizer,
        dataset_path=os.path.join(config.output_dir, config.log_probs_output_filename),
        output_dir=config.output_dir,
        output_filename=config.evaluation_output_filename,
        icl_examples=config.icl_examples,
        )
        
    else:
        get_multi_token_log_probs(
            model=model,
            tokenizer=tokenizer,
            dataset_path=config.dataset_path,
            output_dir=config.output_dir,
            output_filename=config.log_probs_output_filename,
            is_user_prompt=config.is_user_prompt,
        )
    
        evaluate_model_on_multi_token_completion(
            model=model,
            tokenizer=tokenizer,
            dataset_path=os.path.join(config.output_dir, config.log_probs_output_filename),
            output_dir=config.output_dir,
            output_filename=config.evaluation_output_filename,
            icl_examples=config.icl_examples,
        )
        
    compute_metrics(
        predictions_path=os.path.join(config.output_dir, config.evaluation_output_filename),
        output_dir=config.output_dir,
        output_filename=config.metrics_output_filename,
    )
    
    
if __name__ == "__main__":
    config = ExperimentConfig()
    run_experiment(config)