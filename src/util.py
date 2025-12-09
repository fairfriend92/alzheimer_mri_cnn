from pathlib import Path
from collections import Counter
import itertools
import ast

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

def check_sampler(train_loader, n_batches=200):
    cnt = Counter()
    for i, (_, y) in enumerate(train_loader):
        cnt.update([int(v) for v in y])
        if i + 1 >= n_batches:
            break
    return cnt
