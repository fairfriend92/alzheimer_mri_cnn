import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Used to plot confusion matrix
from datetime import datetime # Used to name the figure folder
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

# Check if batch is balanced after sampling
def check_sampler(train_loader, n_batches=200):
    cnt = Counter()
    for i, (_, y) in enumerate(train_loader):
        cnt.update([int(v) for v in y])
        if i + 1 >= n_batches:
            break
    return cnt

# Plot figures such as ROC, confusion matrix...
def plot_figs(ys, ps, probs, auc_score, figs_paths='./outputs/figs'):
  timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
  final_path = f'{figs_paths}/{timestamp}'
  os.makedirs(final_path, exist_ok=True)

  # Plot and save confusion matrix 
  cm = confusion_matrix(ys, ps)
  cm_path = f'{final_path}/cm.png'
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
  plt.savefig(f'{final_path}/ROC.png', dpi=300, bbox_inches='tight')
  plt.close()
  print(f"Saved ROC to: {final_path}/ROC.png")
