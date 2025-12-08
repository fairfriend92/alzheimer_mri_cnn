import numpy as np
import joblib   # Used to save models 
import matplotlib.pyplot as plt
import seaborn as sns   # Used to plot heatmap
import pandas as pd
import json # Used to access patients dataset 
import torchio as tio # Used to trasnform MRI volumes 
import argparse
from collections import Counter
import os
import sys 

from preprocessing import preprocess_all
from util import update_args_from_file
from neural_networks import Simple3DCNN, Complex3DCNN, Medium3DCNN

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler


class OasisDataset(Dataset):
    def __init__(self, volumes, labels, transform=None):
        self.volumes = volumes
        self.labels = labels
        self.transform = transform
        self.class_counts = [labels.count(0), labels.count(1)]
        self.weights = 1. / torch.tensor(self.class_counts, dtype=torch.float)
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
        
def train_evaluate(train_loader, test_loader, nn_type='complex'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to device
    if nn_type == 'simple': 
      model = Simple3DCNN().to(device)
    elif nn_type == 'complex':
      model = Complex3DCNN().to(device)
    elif nn_type == 'medium':
      model = Medium3DCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Update parameters with backpropagation. lr: learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 10
    
    print("Training...")
    
    for epoch in range(n_epochs):        
        model.train() # Enable train mode (useful for dropout and batch normalization)
        total_loss = 0

        for vol, label in train_loader:
            # Move batch to device 
            vol = vol.to(device)
            label = label.to(device)
            
            ''' Forward pass '''

            # Nullify parameters' gradients to avoid accumulation
            optimizer.zero_grad()
            
            # Get logits, i.e. tensor with outputs of last layer 
            # to be passed to an activation function to get probabilities
            output = model.forward(vol)
            
            # Compute CrossEntropyLoss (softmax is done internally)
            loss = criterion(output, label)
            
            ''' Backward pass '''
            
            # Compute parameters' gradients with respect to loss 
            loss.backward()
            
            # Update parameters 
            optimizer.step()
            
            # Get float value from loss tensor 
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss = {total_loss/len(train_loader):.4f}")
        
    print("Evaluating...")
        
    model.eval() # Enable evaluation mode (disable dropout and batch normalization)
    correct = 0
    total = 0

    with torch.no_grad(): # Stop computation of gradients to avoid memory waste 
        for vol, label in test_loader:
            vol = vol.to(device)
            label = label.to(device)

            output = model(vol)
            
            # Choose classes with highes logit 
            pred = output.argmax(dim=1)
            
            print("Output:", output)
            print("Pred:", pred)
            print("Label:", label)

            correct += (pred == label).sum().item()
            total += label.size(0)

    print("Accuracy:", correct / total)

if __name__ == "__main__": 
    ''' Parse arguements '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net_type', type=str, default=None, 
                        help='Type of network to use')
    parser.add_argument('-t', '--transform', action='store_true', 
                        help='Transform training data')
    parser.add_argument('-s', '--sampler', action='store_true', 
                        help='Sample training data')
    parser.add_argument('-d', '--discs', type=int, default=None, 
                        help='Number of OASIS discs to download')
    parser.add_argument('-i', '--input', type=str, default=None, 
                        help='Read input file')
    args = parser.parse_args()

    # Read arguements from input file
    if args.input is not None:
      print(f"Reading arguements from file {args.input}.txt")
      update_args_from_file(args)

    # Read number of discs
    if args.discs is None:
        print("No --discs argument, default=5 used.")
        args.discs = 5
    if args.discs > 12 or args.discs < 1:
        print("OASIS has 12 discs. 5 discs will be downloaded.") 
        args.discs = 5

    # Trasnformation to apply online during training.
    # (In each epoch the transformation changes slightly).
    my_transform = None     
    if args.transform:
      print("Using transformation on training dataset.")
      my_transform = tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.1), degrees=5),
        tio.RandomNoise(mean=0, std=0.01)
      ])
    else:
        print("No transformation will be performed on the training dataset.")

    # Read neural network model to be used 
    incorrect_type = (args.net_type != 'simple' and 
                      args.net_type != 'complex' and 
                      args.net_type != 'medium')
    if args.net_type is None or incorrect_type:
        print("No network type specified.")
        args.net_type = 'complex'
    print(f"Using {args.net_type} network type.")

    ''' Preprocess dataset '''

    preprocess_all(n_discs=args.discs)
      
    ''' Prepare dataset and start training '''  
    
    # Create torch dataset  
    X, y = load_dataset("./data/processed", "./data/processed/dataset.json")
    dataset = OasisDataset(X, y, my_transform)
    
    # Split dataset into train and test 
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # Balance num of samples for each class
    if args.sampler:
      print("Using sampling on training dataset.")
      train_indices = train_ds.indices  
      train_sample_weights = [dataset.sample_weights[i] for i in train_indices]
      
      train_sampler = WeightedRandomSampler(
          weights=train_sample_weights,
          num_samples=len(train_sample_weights),
          replacement=True
      )

      train_loader = DataLoader(train_ds, batch_size=2, shuffle=True) 
    else:
        print("No sampling will be performed on the training dataset.")
    
    # Load train a test dataset 
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True) 
    test_loader  = DataLoader(test_ds, batch_size=2)  

    # Check if training dataset i balanced
    train_labels = [int(dataset[i][1]) for i in train_ds.indices]  # dataset[i] = (volume, label)
    counter = Counter(train_labels)
    print(f'Train labels:{counter}')

    train_evaluate(train_loader, test_loader, args.net_type)
