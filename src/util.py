import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Used to plot confusion matrix
import csv  # Used to save metrics in human readable form
import json # Used to access metrics in file
import itertools
import ast
import os
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix, RocCurveDisplay

# Read arguements from file
def update_args_from_file(args):   
    file_path = Path("./inputs") / f"{args.input}.txt"
    try:
      with open(file_path, "r") as f:
          for line in f:
              line = line.strip()
              if not line or line.startswith("#"):
                  continue
              parts = line.split(None, 1)
              if len(parts) != 2:
                  continue
              key, value_str = parts
              if hasattr(args, key):
                  try:
                      value = ast.literal_eval(value_str)
                  except (ValueError, SyntaxError):
                      value = value_str
                  setattr(args, key, value)
    except FileNotFoundError:
      print(f"{file_path} does not exist. Using default arguements.")
    return args

# Save arguements to file
def save_args_to_file(args, timestamp, outputs_path='./outputs'):
  final_path = f'{outputs_path}/{timestamp}'
  os.makedirs(final_path, exist_ok=True)

  with open(f"{final_path}/params.txt", "w") as f:
      for key, value in vars(args).items():
          f.write(f"{key} {value}\n")

# Check if batch is balanced after sampling
def check_sampler(train_loader, n_batches=200):
    cnt = Counter()
    for i, (_, y) in enumerate(train_loader):
        cnt.update([int(v) for v in y])
        if i + 1 >= n_batches:
            break
    return cnt

# Plot figures such as ROC, confusion matrix...
def plot_figs(ys, ps, probs, auc_score, timestamp, outputs_path='./outputs', fold=None):
  final_path = f'{outputs_path}/{timestamp}/figs'
  os.makedirs(final_path, exist_ok=True)

  # Plot and save confusion matrix 
  cm = confusion_matrix(ys, ps)
  if fold is None:
    cm_path = f'{final_path}/cm.png'
  else:
    cm_path = f'{final_path}/cm_fold{fold}.png'
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.savefig(cm_path)
  plt.close()
  print(f"Saved confusion matrix to: {cm_path}")

  # Plot ROC curve
  fig, ax = plt.subplots(figsize=(8, 6))
  RocCurveDisplay.from_predictions(ys, probs, ax=ax)
  ax.set_title(f'ROC Curve (AUC = {auc_score:.3f})')
  if fold is None:
    ROC_path = f'{final_path}/ROC.png'
  else:
    ROC_path = f'{final_path}/ROC_fold{fold}.png'
  plt.savefig(ROC_path, dpi=300, bbox_inches='tight')
  plt.close()

# Compute the average metrics of K fold stratification and save them to file 
def compute_avg_fold_metrics(all_folds_metrics, timestamp, outputs_path='./outputs'):
  final_path = f'{outputs_path}/{timestamp}/metrics'
  os.makedirs(final_path, exist_ok=True)

  avg_accuracy = sum(m["accuracy"] for m in all_folds_metrics) / len(all_folds_metrics)
  avg_auc      = sum(m["auc"] for m in all_folds_metrics) / len(all_folds_metrics)

  avg_f1_macro = sum(m["classification_report"]["macro avg"]["f1-score"]
                    for m in all_folds_metrics) / len(all_folds_metrics)

  avg_metrics = {
      "avg_accuracy": avg_accuracy,
      "avg_auc": avg_auc,
      "avg_f1_macro": avg_f1_macro
  }

  print("\n=== AVERAGED METRICS ===")
  print(avg_metrics)

  with open(f"{final_path}/fold_metrics.json", "w") as f:
    json.dump(all_folds_metrics, f, indent=2)

  with open(f"{final_path}/avg_metrics.json", "w") as f:
    json.dump(avg_metrics, f, indent=2)

  with open(f"{final_path}/fold_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "accuracy", "auc", "f1_macro"])

    for m in all_folds_metrics:
        f1_macro = m["classification_report"]["macro avg"]["f1-score"]
        writer.writerow([m["fold"], m["accuracy"], m["auc"], f1_macro])

    # Write averages row
    writer.writerow(["AVERAGE", avg_accuracy, avg_auc, avg_f1_macro])  



