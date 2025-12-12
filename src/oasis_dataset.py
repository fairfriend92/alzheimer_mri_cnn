import sys 
import os
import json # Used to access patients dataset 
import numpy as np
import torch
from torch.utils.data import Dataset

class OasisDataset(Dataset):
    def __init__(self, volumes, labels, transform=None):
        self.volumes = volumes
        self.labels = labels
        self.transform = transform
        self.class_counts = [labels.count(0), labels.count(1)]
        self.counts = torch.tensor(self.class_counts, dtype=torch.float)
        self.weights = 1. / self.counts
        self.norm_weights = self.counts.sum() / self.counts # Normalized weights
        self.sample_weights = [self.weights[label] for label in labels]

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        vol = torch.tensor(self.volumes[idx], dtype=torch.float32)

        if self.transform:
            vol = self.transform(vol)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return vol, label
        
def load_dataset(processed_dir, dataset_file):
    print("Loading dataset...")
    
    try:
        with open(dataset_file) as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: {dataset_file} not found")
        sys.exit(1)  

    X = []
    y = []

    for dict in dataset:
        subject_id = dict["id"]        
        npy_file = os.path.join(processed_dir, f"{subject_id}.npy")
        try:
            volume = np.load(npy_file)
        except FileNotFoundError:
            print(f"Error: {npy_file} not found")
            sys.exit(1)        
        X.append(volume)
        y.append(dict["label"])
    
    #X = np.array(X)  
    #y = np.array(y)  
    return X, y
        