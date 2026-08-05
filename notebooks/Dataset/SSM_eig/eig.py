from pathlib import Path
import ast
from tqdm.notebook import tqdm
from natsort import natsorted
import os
import numpy as np
import json

base_dir = './notebooks/Dataset/SSM_eig/artifact/Small_lr_cJhJ'  # Change this to your actual path
all_complex_lists = []



for json_path in tqdm(natsorted(list(Path(base_dir).glob("*.json")))):  # from 0 to 999
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            complex_strs = data["data"]  # list of lists with strings
            complex_numbers = [complex(ast.literal_eval(s[0])) for s in complex_strs]
            all_complex_lists.append(complex_numbers)
    else:
        print(f"Missing: {json_path}")

# Now `all_complex_lists` is a list of lists of complex numbers

all_complex_lists = np.array(all_complex_lists)

print(all_complex_lists)