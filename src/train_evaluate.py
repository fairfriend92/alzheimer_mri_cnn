import numpy as np
import joblib   # Used to save models 
import matplotlib.pyplot as plt
import seaborn as sns   # Used to plot heatmap
import pandas as pd
import json # Used to access patients dataset 
import os
import sys 

from data_download import download_oasis
from preprocessing import preprocess_all

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class Simple3DCNN(nn.Module):
    def __init__(self, input_shape=(128,128,128)):
        super().__init__()
        D,H,W = input_shape
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64*(D//4)*(H//4)*(W//4), 128)
        self.fc2 = nn.Linear(128, 2)  # output: 2 classes 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class OasisDataset(Dataset):
    def __init__(self, volumes, labels, transform=None):
        self.volumes = volumes
        self.labels = labels
        self.transform = transform

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
        
def train_evaluate(train_loader, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Simple3DCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 10
    
    print("Training...")
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for vol, label in train_loader:
            vol = vol.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(vol)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss = {total_loss/len(train_loader):.4f}")
        
    print("Evaluating...")
        
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for vol, label in test_loader:
            vol = vol.to(device)
            label = label.to(device)

            output = model(vol)
            pred = output.argmax(dim=1)

            correct += (pred == label).sum().item()
            total += label.size(0)

    print("Accuracy:", correct / total)

if __name__ == "__main__": 
    my_n_discs = None
    if len(sys.argv) > 1:
        my_n_discs = int(sys.argv[1])
        my_n_discs = min(12, my_n_discs) # Oasis has only 12 discs of data 
        
    if my_n_discs is not None and my_n_discs > 0:    
        download_oasis(n_discs=my_n_discs)
        preprocess_all(n_discs=my_n_discs)
    
    # Create torch dataste 
    X, y = load_dataset("./data/processed", "./data/processed/dataset.json")
    dataset = OasisDataset(X, y)
    
    # Split dataset into train and test 
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=2)

    train_evaluate(train_loader, test_loader)
