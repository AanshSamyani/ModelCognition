import os
import json
import torch

from tqdm import tqdm
from dataclasses import dataclass
from utils import find_assistant_token_end
from get_log_probs import get_single_token_log_probs
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation import evaluate_model_on_single_token_completion

@dataclass
class ExperimentConfig:
    model_name: str = "/nlsasfs/home/isea/isea10/aansh/deception_detection/weights/Llama-3.1-8B-Instruct"
    dataset_path: str = "/nlsasfs/home/isea/isea10/aansh/introspection/data/exp1/single_token_strict.json"
    output_dir: str = "/nlsasfs/home/isea/isea10/aansh/introspection/results/exp1_llama_8b"
    log_probs_output_filename: str = "single_token_strict_log_probs.json"
    evaluation_output_filename: str = "single_token_strict_evaluation.json"
    icl_examples: int = 6 
    
def run_experiment(config: ExperimentConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
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
    
if __name__ == "__main__":
    config = ExperimentConfig()
    run_experiment(config)