import os
import json

from datasets import Dataset

class MultiTokenCompletionDataset(Dataset):
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "messages": item["messages"]
        }
        
