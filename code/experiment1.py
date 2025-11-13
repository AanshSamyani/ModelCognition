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
    dataset_path: str = ""
    output_dir: str = ""
    log_probs_output_filename: str = ""
    evaluation_output_filename: str = ""
    icl_examples: int = 6 