from pathlib import Path
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
