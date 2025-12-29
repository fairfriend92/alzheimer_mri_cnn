''' General imports '''
import sys 
import numpy as np
import joblib   # Used to save models 
import matplotlib.pyplot as plt
import seaborn as sns   # Used to plot heatmap
import pandas as pd
import torchio as tio # Used to trasnform MRI volumes 
import argparse
import sqlite3
from collections import Counter
from collections import defaultdict # Dict that stores probs for each patient ID
from pathlib import Path
from datetime import datetime # Used to name the output folder

''' My imports '''
from oasis_dataset import OasisDataset, load_dataset, augment_dataset
from preprocessing import preprocess_all
from util import (update_args_from_file, check_sampler, 
                  plot_figs, compute_avg_fold_metrics, save_args_to_file)
from neural_networks import Simple3DCNN, Complex3DCNN, Medium3DCNN

''' ML imports '''
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, random_split, 
                              WeightedRandomSampler, Subset)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

def train_evaluate(train_loader, test_loader, dataset, args, timestamp, fold=None,
                   db_path = "data/processed/oasis.db"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move class weights to device (OASIS dataset is not balanced)
    class_weights = dataset.norm_weights.to(device)
    
    # Move model to device
    if args.net_type == 'simple': 
      model = Simple3DCNN().to(device)
    elif args.net_type == 'complex':
      model = Complex3DCNN().to(device)
    elif args.net_type == 'medium':
      model = Medium3DCNN().to(device)
    
    if args.sampler:
      criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
      criterion = nn.CrossEntropyLoss()
    
    # Update parameters with backpropagation. lr: learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 10
    
    print("Training...")
    
    for epoch in range(n_epochs):        
        model.train() # Enable train mode (useful for dropout and batch normalization)
        total_loss = 0

        for vol, label, patient_id in train_loader:
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

    # Enable evaluation mode (disable dropout and batch normalization)    
    model.eval() 

    # Dictionary of lists that store probs for each patient id to do test-time augmentation (TTA)
    patient_probs = defaultdict(list)

    # Dictionary to store label for each patient id 
    patient_labels = {}

    with torch.no_grad(): # Stop computation of gradients to avoid memory waste 
        for vol, label, patient_id in test_loader:
            vol = vol.to(device)
            label = label.to(device)
            output = model(vol)

            # Compute prob for Alzheimer's Disease and predict class
            probs_ad = torch.softmax(output, dim=1)[:, 1]  
        
            # Loop over batch to handle different patient_ids
            for i in range(vol.shape[0]):  # Default batch_size=2
                prob_ad = probs_ad[i].item()  # Single scalar now
                pat_id = patient_id[i]        # Single patient_id
                pat_label = label[i].item()   # Single label
                
                # Save in the dictionary the prob and label for this patient
                patient_probs[pat_id].append(prob_ad)
                patient_labels[pat_id] = pat_label

    # Dictionaries where mean TTA probs and preds are saved
    final_probs, final_preds = {}, {}

    # Compute mean TTA probs and preds
    for patient_id, probs in patient_probs.items():
        mean_prob = sum(probs) / len(probs)
        final_probs[patient_id] = mean_prob
        final_preds[patient_id] = int(mean_prob > 0.5)

    # Connect to database to save predictions and probabilities
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Counters for correct and total predictions
    correct, total = 0, 0

    # Lists for all labels, preds and probs  
    labels, preds, probs = [], [], []

    for pid in final_probs:
      # Count correct predictions
      if final_preds[pid] == patient_labels[pid]:
            correct += 1
      total += 1

      # Store label, prob and pred
      labels.append(patient_labels[pid])
      probs.append(final_probs[pid])
      preds.append(final_preds[pid])

      # Save to database 
      cursor.execute("""
          INSERT OR REPLACE INTO predictions
          (patient_id, model_name, prob_ad, predicted_label)
          VALUES (?, ?, ?, ?)
      """, (pid, f"{args.net_type}_cnn", final_probs[pid], final_preds[pid]))

    # Compute accuracy
    accuracy = correct / total

    # Close connection to database 
    conn.commit()
    conn.close() 

    # Print metrics 
    report = classification_report(labels, preds, digits=3, output_dict=True)
    auc_score = roc_auc_score(labels, probs)
    print(f"Accuracy:{accuracy}")
    print(f"AUC: {auc_score:.3f}")
    print(classification_report(labels, preds, digits=3))    

    # Plot figures
    plot_figs(labels, preds, probs, auc_score, timestamp, fold=fold)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "auc": auc_score,
        "fold": -1 if fold is None else fold+1
    }

def launch_training(args, train_ds, test_ds, dataset, timestamp, fold=None):
  # Balance num of samples for each class and load train dataset
  if args.sampler:
    print("\nUsing sampling on training dataset.")
    train_indices = train_ds.indices  
    train_sample_weights = [dataset.sample_weights[i] for i in train_indices]
    
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=2, sampler=train_sampler, shuffle=False) 
  else:
    print("\nNo sampling will be performed on the training dataset.")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
  
  # Load test dataset      
  test_loader  = DataLoader(test_ds, batch_size=2)  

  # Check counts for each class if sampler is used
  if args.sampler:
    train_labels = [int(dataset[i][1]) for i in train_ds.indices]  # dataset[i] = (volume, label)
    counter = Counter(train_labels)
    print(f'Train labels:{counter}')

    batch_cnt = check_sampler(train_loader)
    print(f'Batch labels:{batch_cnt}')

  return train_evaluate(train_loader, test_loader, dataset, args, timestamp, fold)

if __name__ == "__main__": 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")

    ''' Parse arguements '''    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net_type', type=str, default=None, 
                        help='Type of network to use')
    parser.add_argument('-t', '--transform', action='store_true', 
                        help='Transform training data')
    parser.add_argument('-a', '--augment', type=int, default=None, 
                        help='Augment minor class with trasformations')
    parser.add_argument('-at', '--aug_test', action='store_true', 
                        help='Augment test dataset')
    parser.add_argument('-k', '--kfolds', type=int, default=None, 
                        help='Use stratified K fold')
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

    # Save arguements to file in output folder for safekeeping 
    save_args_to_file(args, timestamp)

    if args.sampler and args.augment is not None:
      print("Cannot sample and augment at the same time, exiting.")
      sys.exit()

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
    volumes, labels, patients_ids = load_dataset("./data/processed", 
                                   "./data/processed/dataset.json")    
    dataset = OasisDataset(volumes, labels, patients_ids, my_transform)

    # Split dataset into train and test...   
    if args.kfolds is None or args.kfolds <= 0:
      train_size = int(0.8 * len(dataset))
      test_size  = len(dataset) - train_size
      train_ds, test_ds = random_split(dataset, [train_size, test_size])

      train_idx = train_ds.indices
      test_idx  = test_ds.indices

      # Augment dataset if needed
      if args.augment and args.augment > 0:
        aug_ds = augment_dataset(volumes, labels, patients_ids, train_idx, test_idx, 
                                 args.augment, my_transform, args.aug_test)
        train_ds = aug_ds[0]
        if args.aug_test: test_ds = aug_ds[1]

      launch_training(args, train_ds, test_ds, dataset, timestamp)
    #...or use stratified K fold
    else:
      print(f"\nUsing stratified k folds with k={args.kfolds}.")
      skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)
      all_folds_metrics = []
      for fold, (train_idx, test_idx) in enumerate(skf.split(volumes, labels)):
          print(f"\n===== FOLD {fold+1} =====", flush=True)

          train_ds = Subset(dataset, train_idx)
          test_ds  = Subset(dataset, test_idx)
          
          if args.augment and args.augment > 0:
            aug_ds = augment_dataset(volumes, labels, patients_ids, train_idx, test_idx, 
                                     args.augment, my_transform, args.aug_test)
            train_ds = aug_ds[0]
            if args.aug_test: test_ds = aug_ds[1]
            
          fold_metrics = launch_training(args, train_ds, test_ds, dataset, timestamp, fold)
          all_folds_metrics.append(fold_metrics)
      compute_avg_fold_metrics(all_folds_metrics, timestamp)

    
