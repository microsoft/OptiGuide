import glob
import json
import os
import pdb
import sys
import argparse
import pickle
import re
from collections import defaultdict

import numpy as np



############# helper functions
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

    z = re.findall("(\d+)", path)
    if z:
        return int(z[0])

    raise ValueError("Cannot find the MILP ID for " + path)
    
def add_data(filename, text, train_ids):
    global TRAIN_DATA, TEST_DATA
    _id = milp_id(filename)
    if _id in train_ids:
        TRAIN_DATA[filename].append(text)
    else:
        TEST_DATA[filename].append(text)

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
        # pdb.set_trace()
    return ans
####### 



### aggregate data and description
def build_dataset(parent_data_dir, parent_desc_dir, multimodal_data_file, desc_suffix=""):
    desc_path_glob = os.path.join(parent_desc_dir, f"*/desc_*{desc_suffix}.txt")

    descs = glob.glob(desc_path_glob)

    count = 0
    data = []
    for desc in descs:
        gz_path = desc.replace(parent_desc_dir, parent_data_dir).replace("desc", "data").replace(desc_suffix, "").replace(".txt", ".pkl.gz")
        
        if not os.path.exists(gz_path):
            continue
                
        count += 1
        data.append({
            "id": str(count), "image": gz_path, "text_path": desc,
            "conversations": [{
                "from": "human",
                "value": "<image>\nDescribe the data."
            }, {
                "from": "gpt",
                "value": open(desc, "r").read()
            }]
        })

    json.dump(data, open(multimodal_data_file, "w"), indent=2)


### split data into train/test/val
def split_data(multimodal_data_file, parent_data_dir, parent_code_dir, parent_save_dir, train_ids, out_suffix=""):
    global TRAIN_DATA, TEST_DATA
    TRAIN_DATA = defaultdict(list)
    TEST_DATA = defaultdict(list)

    multimodal_data = json.load(open(multimodal_data_file, "r"))
    multimodal_files = [item["image"] for item in multimodal_data]
    multimodal_desc_files = [item["text_path"] for item in multimodal_data]
    # remove files that does not exist
    x = zip(multimodal_files, multimodal_desc_files)
    x = [item for item in x if os.path.exists(item[0]) and os.path.exists(item[1])]
    multimodal_files, multimodal_desc_files = zip(*x)

    for i, (mps_file, desc_file) in enumerate(zip(multimodal_files, multimodal_desc_files)):
        add_data(mps_file, open(desc_file, "r").read(), train_ids=train_ids)

        class_name = os.path.basename(os.path.dirname(desc_file))
        code_filename = os.path.join(parent_code_dir, f"{class_name}.py")

        if os.path.exists(code_filename):
            for component in parse_code(code_filename):
                add_data(mps_file, component, train_ids=train_ids)


    if parent_data_dir:
        for problem_dir in glob.glob(os.path.join(parent_data_dir, "*")):
            _id = milp_id(problem_dir)
            src_codename = glob.glob(os.path.join(parent_code_dir, f"milp_{_id}-*.py"))[0]
            code_components = parse_code(src_codename)

            for mps_filename in glob.glob(os.path.join(problem_dir, "*.pkl.gz")):
                for component in code_components:
                    add_data(mps_filename, component, train_ids=train_ids)

    # Finally, dump the data. Let's mainly use the pickle format because of its compression
    json.dump(TRAIN_DATA, open(os.path.join(parent_save_dir, f"train_{out_suffix}data.json"), "w"), indent=2)
    json.dump(TEST_DATA, open(os.path.join(parent_save_dir, f"test_{out_suffix}data.json"), "w"), indent=2)

    pickle.dump(TRAIN_DATA, open(os.path.join(parent_save_dir, f"train_{out_suffix}data.pkl.gz"), "wb"))
    pickle.dump(TEST_DATA, open(os.path.join(parent_save_dir, f"test_{out_suffix}data.pkl.gz"), "wb"))    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_code_dir", type=str, default="milp_code_v1/code")
    parser.add_argument("--parent_data_dir", type=str, default="save_dir/contrast/data")
    parser.add_argument("--parent_desc_dir", type=str, default="save_dir/contrast/conv")
    parser.add_argument("--parent_save_dir", type=str, default="save_dir/contrast")
    parser.add_argument("--multimodal_data_file", type=str, default="save_dir/contrast/data.json")
    parser.add_argument("--desc_suffix", type=str, default="")
    parser.add_argument("--out_suffix", type=str, default="ours")

    args = parser.parse_args()

    build_dataset(args.parent_data_dir, args.parent_desc_dir, multimodal_data_file=args.multimodal_data_file,
                  desc_suffix=args.desc_suffix)


    # First, randomly determine the IDs for trainning and testing
    total_length = 10000 # assume we have 10000 classes, which is above the actual number
    train_ids = np.random.choice(total_length, 8000, replace=False)
    test_ids = [i for i in range(total_length) if i not in train_ids]

    split_data(multimodal_data_file = args.multimodal_data_file,
               parent_data_dir = args.parent_data_dir,
               parent_code_dir=args.parent_code_dir, 
               parent_save_dir=args.parent_save_dir, 
               train_ids=train_ids, 
               out_suffix=args.out_suffix)