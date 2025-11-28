import os
import json

from torch.utils.data import Dataset

class MultiTokenCompletionDataset(Dataset):
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "completion_1": item["completion_1"],
            "completion_2": item["completion_2"],
            "logit_1": item["logit_1"],
            "logit_2": item["logit_2"]
        }