import glob
import json
import os
import random
import argparse

random.seed(1)

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Process data split.")
parser.add_argument("--parent_data_dir", type=str, default='save_dir/gap_data', help="Directory for parent data.")
parser.add_argument("--parent_data_metadata_dir", type=str, default='save_dir/gap_data/metadata', help="Directory for data metadata.")
parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")
parser.add_argument("--difficulty_levels", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty levels to consider.")

args = parser.parse_args()

ROOT_DIR = os.path.join(args.parent_data_dir, args.code_str)

difficulty_levels = args.difficulty_levels
SEED_DIRS = [os.path.join(args.parent_data_dir, f"milp_{i}-{difficulty}") for i in range(8) for difficulty in difficulty_levels]

SPLIT_OUTPUT_DIR = os.path.join(args.parent_data_metadata_dir, args.code_str)
####################################################################################

os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)

def load_gap_data(directory):
    gap_data = []
    for data_file in glob.glob(os.path.join(directory, "data_*.pkl.gz")):
        label_file = data_file.replace("data_", "label_").replace(".pkl.gz", ".json")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                try:
                    label = json.load(f)
                except:
                    continue
                if 'final_gap' in label and label['final_gap'] < 0.01:
                    gap_data.append(data_file)
    return gap_data


def rel_path(x, parent_dir=PARENT_DIR):
    return os.path.relpath(x, parent_dir)


def get_problem_ids(x):
    return int(os.path.basename(x).split("-")[0].split("_")[1])


def sort_directory(x):
    if 'ecole-gasse' in x: return 0

    parts = x.split('/')
    index = int(parts[-2].split('_')[1].split('-')[0])
    return index

def default_data_split():
    ### MILP-Evolve (Ours)
    milp_problems = glob.glob(os.path.join(ROOT_DIR, "milp_*"))
    milp_problem_ids = [get_problem_ids(x) for x in milp_problems]
    milp_problem_ids = list(set(milp_problem_ids)) # remove duplicates

    # Split the problem ids into train and test
    random.shuffle(milp_problem_ids)
    n_train_cls = int(len(milp_problem_ids)*0.8)
    print(f"Ours: N train Python classes: {n_train_cls}")
    train_problem_ids = milp_problem_ids[:n_train_cls]
    test_problem_ids = milp_problem_ids[n_train_cls:]

    # check no train and test ids overlap
    assert len(set(train_problem_ids).intersection(set(test_problem_ids))) == 0
    
    test_mps_nested = [load_gap_data(os.path.join(ROOT_DIR, f"milp_{x}-*")) for x in test_problem_ids]
    test_mps = [x for gap_data in test_mps_nested for x in gap_data]

    mps_in_train_cls_nested = [load_gap_data(os.path.join(ROOT_DIR, f"milp_{x}-*")) for x in train_problem_ids]
    mps_in_train_cls = [x for gap_data in mps_in_train_cls_nested for x in gap_data]

    # Split the train problems into train and validation
    random.shuffle(mps_in_train_cls)
    n_train_mps = int(len(mps_in_train_cls)*0.8)
    print(f"N train MPS: {n_train_mps}")
    train_mps = mps_in_train_cls[:n_train_mps]
    val_mps = mps_in_train_cls[n_train_mps:]

    data = {
        "train": sorted([rel_path(x) for x in train_mps], key=sort_directory),
        "val": sorted([rel_path(x) for x in val_mps], key=sort_directory),
        "test": sorted([rel_path(x) for x in test_mps], key=sort_directory)
    }

    if not os.path.exists(os.path.join(SPLIT_OUTPUT_DIR, "gap_data_split.json")):
        json.dump(data, open(os.path.join(SPLIT_OUTPUT_DIR, "gap_data_split.json"), "w"), indent=2)

    ### SEED
    seed_milp_problems = [glob.glob(os.path.join(SEED_DIR, "*.pkl.gz")) for SEED_DIR in SEED_DIRS]
    seed_milp_problems = [x for sublist in seed_milp_problems for x in sublist]

    # 80% for train and rest for eval
    random.shuffle(seed_milp_problems)
    n_train_seed = int(len(seed_milp_problems)*0.8)
    print(f"Seed: N train Python classes: {n_train_seed}")
    seed_train_mps = seed_milp_problems[:n_train_seed]
    seed_val_mps = seed_milp_problems[n_train_seed:]

    data = {
        "train": sorted([rel_path(x) for x in seed_train_mps], key=sort_directory),
        "val": sorted([rel_path(x) for x in seed_val_mps], key=sort_directory),
        "test": sorted([rel_path(x) for x in test_mps], key=sort_directory)
    }

    seed_str_list = []
    for diff_str, difficulty in [("E", "easy"), ("M", "medium"), ("H", "hard")]:
        if difficulty in difficulty_levels:
            seed_str_list.append(diff_str)
    seed_str = "_" + "".join(seed_str_list)

    if not os.path.exists(os.path.join(SPLIT_OUTPUT_DIR, f"gap_data_split_train_seed{seed_str}.json")):
        json.dump(data, open(os.path.join(SPLIT_OUTPUT_DIR, f"gap_data_split_train_seed{seed_str}.json"), "w"), indent=2)



if __name__ == '__main__':
    default_data_split()
