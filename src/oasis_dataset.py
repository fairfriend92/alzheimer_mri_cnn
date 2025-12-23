import sys 
import os
import json # Used to access patients dataset 
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset
from pathlib import Path

class OasisDataset(Dataset):
    def __init__(self, volumes, labels, patients_ids, transform=None):
        self.volumes = volumes
        self.labels = labels
        self.transform = transform
        self.class_counts = [labels.count(0), labels.count(1)]
        self.counts = torch.tensor(self.class_counts, dtype=torch.float)
        self.weights = 1. / self.counts
        self.norm_weights = self.counts.sum() / self.counts # Normalized weights
        self.sample_weights = [self.weights[label] for label in labels]
        self.patients_ids = patients_ids

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        vol = torch.tensor(self.volumes[idx], dtype=torch.float32)

        if self.transform:
            vol = self.transform(vol)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return vol, label, self.patients_ids[idx]
        
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
    subjects_ids = []

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
        subjects_ids.append(subject_id)        
  
    return X, y, subjects_ids

def augment_dataset(volumes, labels, train_idx, test_idx, num_copies, 
                    base_transform=None, aug_test=True):
  if aug_test:
    print("Augmenting training and test datasets with transformation.")
  else:
    print("Augmenting training dataset with transformation.")

  # Create transformation used to augment dataset
  minor_transform = tio.Compose([
      tio.RandomAffine(scales=(0.95,1.05), degrees=10),
      tio.RandomFlip(axes=(0,)),
      tio.RandomElasticDeformation(num_control_points=7, max_displacement=2.0),
      tio.RandomNoise(std=0.02),
      tio.RandomBiasField(),
  ])

  if aug_test:
    aug_idx = [train_idx, test_idx]
  else:
    aug_idx = [train_idx]

  aug_ds = []

  # Apply trasformations to each sample of the minor class
  for idx in aug_idx:
    # Idx of minor class' labels
    minor_idx = [i for i in idx if labels[i] == 1]

    aug_vols = [volumes[i] for i in idx]
    aug_labels = [labels[i]  for i in idx]
    aug_ids = [patients_ids[i] for i in idx]

    done = 0
    for i in minor_idx:
        pct = 100 * done / len(minor_idx) 
        print(f"\rProgress: {pct:6.2f}%", end="")
        vol = torch.tensor(volumes[i], dtype=torch.float32)
        for _ in range(num_copies):
            img = tio.ScalarImage(tensor=vol)
            aug = minor_transform(img)
            aug_vols.append(aug.data.numpy()) # Retrieve numpy
            aug_labels.append(1)
            aug_ids.append(patients_ids[i])

        done = done + 1

    aug_ds.append(OasisDataset(aug_vols, aug_labels, aug_ids, base_transform))

  return aug_ds
        