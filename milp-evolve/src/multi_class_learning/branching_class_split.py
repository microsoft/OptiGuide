import random
from collections import defaultdict
import json
import glob
import os
import copy
import argparse

random.seed(1)


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(description="Process data split.")
parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")
parser.add_argument("--branching_root_dir", type=str, default='save_dir/branching_data', help="Directory for parent data.")
parser.add_argument("--branching_seed_dir", type=str, default='save_dir/branching_data', help="Directory for seed parent data.")
parser.add_argument("--save_instance_dir", type=str, default='save_dir/instances/mps', help="Directory for parent data.")
parser.add_argument("--save_json_dir", type=str, default='save_dir/instances/metadata/by_dir_branching', help="Directory for parent data.")
parser.add_argument("--code_end_idx", type=int, default=100, help="End index for code instances.")
parser.add_argument("--difficulty_levels", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty levels to consider.")
parser.add_argument("--json_suffix", type=str, default='', help="Suffix for the json file.")

args = parser.parse_args()

seed_start_idx=0
seed_end_idx=8
gen_start_idx=8
gen_end_idx=args.code_end_idx
json_suffix = args.json_suffix

BRANCHING_ROOT_DIR = args.branching_root_dir
INSTANCE_ROOT_DIR = args.save_instance_dir

ROOT_DIR = os.path.join(args.branching_root_dir, args.code_str)
SEED_DIR = os.path.join(args.branching_seed_dir, args.code_str)

JSON_DIR = os.path.join(args.save_json_dir, args.code_str)

difficulty_levels = ['easy', 'medium']
####################################################################################


difficulty_suffix_list = []
for diff_str, difficulty in [("E", "easy"), ("M", "medium"), ("H", "hard")]:
    if difficulty in difficulty_levels:
        difficulty_suffix_list.append(diff_str)
difficulty_suffix = "_" + "".join(difficulty_suffix_list)

def rel_directory(x, root_dir=BRANCHING_ROOT_DIR):
    return x.replace(root_dir, '').lstrip('/')

def sort_directory(x):
    if 'ecole-gasse' in x: return 0

    parts = x.split('/')
    index = int(parts[-1].split('_')[1].split('-')[0])
    return index


def default_data_split():
    ################## MILP-Evolve (Ours) data split #######gen_start_idx
    directories = [x for x in glob.glob(f'{ROOT_DIR}/*') if gen_start_idx <= int(x.split('/')[-1].split('_')[1].split('-')[0]) < gen_end_idx and len(glob.glob(f"{x}/*.pkl.gz")) > 0]

    # Extract indices and create a mapping
    index_map = defaultdict(list)
    for directory in directories:
        parts = directory.split('/')[-1].split('_')
        index = int(parts[1].split('-')[0])
        index_map[index].append(directory)

    # Extract indices and shuffle
    indices = list(index_map.keys())

    random.shuffle(indices)

    # Split indices
    num_dirs = len(directories)
    train_split = int(0.7 * num_dirs)
    val_split = int(0.1 * num_dirs)

    train_dirs = []

    val_dirs = []
    test_dirs = []

    # Iterate through the shuffled indices to collect directories until reaching the limit
    count_train = 0
    count_val = 0
    count_test = 0

    for idx in indices:
        if count_train < train_split:
            train_dirs.extend(index_map[idx])
            count_train += len(index_map[idx])
        elif count_val < val_split:
            val_dirs.extend(index_map[idx])
            count_val += len(index_map[idx])
        else:
            test_dirs.extend(index_map[idx])
            count_test += len(index_map[idx])


    # Adjust to make sure the splits are exactly 70%, 10%, 20%
    train_dirs = sorted(train_dirs[:train_split], key=sort_directory)
    val_dirs = sorted(val_dirs[:val_split], key=sort_directory)
    test_dirs = sorted(test_dirs[:num_dirs - train_split - val_split], key=sort_directory)

    # print the number of train, val, test directories
    print(f"# Ours Train: {len(train_dirs)}")
    print(f"# Ours Val: {len(val_dirs)}")
    print(f"# Ours Test: {len(test_dirs)}")
    print(f'# Ours Total: {len(train_dirs) + len(val_dirs) + len(test_dirs)}')

    # Create a joint dictionary
    data_splits = {
        "train": [rel_directory(d) for d in train_dirs],
        "val": [rel_directory(d) for d in val_dirs],
        "test": [rel_directory(d) for d in test_dirs]
    }

    # Save to a JSON file
    json_save_dir = os.path.join(JSON_DIR)
    os.makedirs(json_save_dir, exist_ok=True)

    json_path = os.path.join(json_save_dir, f'instances_split_ours{json_suffix}.json')
    if not os.path.exists(json_path):
        with open(json_path, 'w') as json_file:
            json.dump(data_splits, json_file, indent=4)
        print(f"Data splits saved to {json_path}")
    original_train = copy.deepcopy(data_splits["train"])

    ################## Combine with Seed directories ##################

    all_seed_directories = [x for x in glob.glob(f'{SEED_DIR}/*') if seed_start_idx <= int(x.split('/')[-1].split('_')[1].split('-')[0]) < seed_end_idx and len(glob.glob(f"{x}/*.pkl.gz")) > 0]
    all_seed_directories = [rel_directory(d) for d in sorted([x for x in all_seed_directories if any(diff in x for diff in difficulty_levels)], key=sort_directory)]
    data_splits["train"] = all_seed_directories + original_train
    json_path = os.path.join(json_save_dir, f'instances_split_ours+seed{difficulty_suffix}{json_suffix}.json')
    if not os.path.exists(json_path):
        with open(json_path, 'w') as json_file:
            json.dump(data_splits, json_file, indent=4)
        print(f"Data splits saved to {json_path}")

    num_ours_seed_train = len(data_splits["train"])
    print(f"# Ours+SEED Train: {num_ours_seed_train}")

    data_splits["train"] = all_seed_directories
    json_path = os.path.join(json_save_dir, f'instances_split_seed{difficulty_suffix}{json_suffix}.json')
    if not os.path.exists(json_path):
        with open(json_path, 'w') as json_file:
            json.dump(data_splits, json_file, indent=4)
        print(f"Data splits saved to {json_path}")
    
    num_seed_train = len(data_splits["train"])
    print(f"# SEED Train: {num_seed_train}")


if __name__ == '__main__':
    default_data_split()