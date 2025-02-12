import json
import os
import pickle
import re
import argparse
from collections import defaultdict

import numpy as np

# First, randomly determine the IDs for trainning and testing
total_length = 10000 # assume we have 10000 classes, which is above the actual number
train_ids = np.random.choice(total_length, 8000, replace=False)
test_ids = [i for i in range(total_length) if i not in train_ids]

def add_data(filename, text):
    global TRAIN_DATA, TEST_DATA
    _id = milp_id(filename)
    if _id in train_ids:
        TRAIN_DATA[filename].append(text)
    else:
        TEST_DATA[filename].append(text)

def milp_id(path):
    x = re.findall("milp_(\d+)-", path)
    if x:
        return int(x[0])
    y = re.findall("(\d+)_gpt", path)
    if y:
        return int(y[0])

    x = re.findall("milp-\d+_(\d+)", path) # for seed+vae and seed+param
    if x:
        return int(x[0])

    z = re.findall("(\d+)_algo", path)
    if z:
        return int(z[0])    
    raise ValueError("Cannot find the MILP ID for " + path)
    

####### helper function

def parse_code(code_filename):
    if not os.path.exists(code_filename):
        return []
    code = open(code_filename, "r").read()

    # first, remove the line that has the substring "time.time()"
    filtered_code = [line for line in code.split("\n") if "time.time()" not in line]
    code = "\n".join(filtered_code)

    ans = []

    class_name = re.findall("class (\w+)", code)
    if class_name:
        class_name = class_name[0]

    ans.append(class_name)

    solve_imp = re.findall("def solve\(.*?\):\n(.+)return ", code, re.DOTALL)
    if solve_imp:
        solve_code = solve_imp[0]
        _code = _remove_heading_spaces(solve_code)
        ans.append(_code)
    return ans

def _remove_heading_spaces(solve_code):
    while True:
        lines = solve_code.split("\n")
        # Check if all non-empty lines have leading spaces
        if all(line.startswith("  ") or line == "" or line.startswith("#") for line in lines):
            # Remove two leading spaces from each line
            solve_code = "\n".join([line[2:] if line.startswith("  ") else line for line in lines])
        else:
            break
    return solve_code

####### Now, Load the Data #####

def aggregate_data(multimodal_data_file, 
                   out_dir="save_dir/contrast", out_suffix=""):
    global TRAIN_DATA, TEST_DATA
    TRAIN_DATA = defaultdict(list)
    TEST_DATA = defaultdict(list)

    # First, loading the llava description
    multimodal_data = json.load(open(multimodal_data_file, "r"))
    multimodal_files = [item["milp"] for item in multimodal_data]
    multimodal_desc_files = [item["text_path"] for item in multimodal_data]
    # remove files that does not exist
    x = zip(multimodal_files, multimodal_desc_files)
    x = [item for item in x if os.path.exists(item[0]) and os.path.exists(item[1])]
    multimodal_files, multimodal_desc_files = zip(*x)

    for i, (mps_file, desc_file) in enumerate(zip(multimodal_files, multimodal_desc_files)):
        add_data(mps_file, open(desc_file, "r").read())

        code_filename = re.sub("desc_seed.*.txt", "milp.py", desc_file)
        if code_filename.endswith(".py"):
            for component in parse_code(code_filename):
                add_data(mps_file, component)

    # Finally, dump the data. Let's mainly use the pickle format because of its compression
    json.dump(TRAIN_DATA, os.path.join(out_dir, open(f"train_{out_suffix}_data.json", "w")), indent=2)
    json.dump(TEST_DATA, os.path.join(out_dir, open(f"test_{out_suffix}_data.json", "w")), indent=2)

    pickle.dump(TRAIN_DATA, os.path.join(out_dir, open(f"train_{out_suffix}_data.pkl.gz", "wb")))
    pickle.dump(TEST_DATA, os.path.join(out_dir, open(f"test_{out_suffix}_data.pkl.gz", "wb")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal_data_file", type=str, default="save_dir/contrast/ours_multimodal.json", help="Multimodal data file")
    parser.add_argument("--out_dir", type=str, default="save_dir/contrast", help="Output directory")
    parser.add_argument("--out_suffix", type=str, default="ours_", help="Output suffix")
    args = parser.parse_args()

    aggregate_data(args.multimodal_data_file, args.out_dir, args.out_suffix)