import os
import shutil
import re
import json
from joblib import Parallel, delayed
import sys

sys.path.append("../milp_evolve_llm")
from src.utils import run_and_save_log, get_status_time_and_gap

seed_folder = 'milp_code_seed'
save_suffix = 'v1'
code_suffix = 'v1'
combined_folder = f'milp_code_{save_suffix}'
gen_folder = f'../milp_evolve_llm/output_milp_{code_suffix}/param_search_dir'
difficulty_list = ['easy', 'medium', 'hard']
GAP_TH = 0

combined_folder_code = os.path.join(combined_folder, 'code')
combined_folder_log = os.path.join(combined_folder, 'log')

seed_start_idx = 0  
seed_end_idx = 8
gen_start_idx = 8  
gen_end_idx = float('inf') 
TIMEOUT_DURATION = 120   
SUBPROCESS_TIMEOUT_DURATION = 180 
n_cpus = 24  # Set the number of CPUs to use for parallel processing
TIME_GAP = 5
HARD_TIME_TH = TIMEOUT_DURATION * 2 / 3 
MEDIUM_TIME_TH = TIMEOUT_DURATION / 3  

# Create directories if they do not exist
os.makedirs(combined_folder_code, exist_ok=True)
os.makedirs(combined_folder_log, exist_ok=True)


def extract_solve_time(log_file):
    solve_time = None
    with open(log_file, 'r') as f:
        for line in f:
            solve_time_match = re.search(r"Solving Time \(sec\)\s*:\s*([\d.]+)", line)
            if solve_time_match:
                solve_time = float(solve_time_match.group(1))
                break
    return solve_time


def process_seed_file(current_idx, idx, difficulty):
    src_file_name = f'milp_{idx}-{difficulty}.py'
    dest_file_name = f'milp_{current_idx}-{difficulty}.py'
    src_file = os.path.join(seed_folder, src_file_name)
    dest_file = os.path.join(combined_folder_code, dest_file_name)
    log_file = os.path.join(combined_folder_log, f'milp_{current_idx}-{difficulty}.txt')

    if os.path.isfile(src_file):
        if not os.path.isfile(dest_file):
            shutil.copy(src_file, dest_file)

        if not os.path.isfile(log_file):
            run_and_save_log(dest_file, log_file, timeout_duration=TIMEOUT_DURATION, subprocess_timeout_duration=SUBPROCESS_TIMEOUT_DURATION)

            # check the gap if gap > 0 remove
            status, time, gap = get_status_time_and_gap(open(log_file, 'r').read())
            print(f'The gap is {gap} for {src_file_name}')
            if gap is None or gap > GAP_TH:
                print(f'Gap is {gap} for {src_file_name}, remove the file')
                os.remove(dest_file)
                os.remove(log_file)


    print(f'Finished processing {src_file_name}')

# First, copy all seed .py files to the combined_folder_code and generate logs
Parallel(n_jobs=n_cpus)(delayed(process_seed_file)(current_idx, idx, difficulty) for current_idx, idx in enumerate(range(seed_start_idx, seed_end_idx)) for difficulty in difficulty_list)

file_mapping = {}


current_idx = 0
for idx in range(seed_start_idx, seed_end_idx):
    for difficulty in difficulty_list:
        src_file_name = f'milp_{idx}-{difficulty}.py'
        dest_file_name = f'milp_{current_idx}-{difficulty}.py'
        tmp_dest_file_path = os.path.join(combined_folder_code, src_file_name)
        src_file = os.path.join(seed_folder, src_file_name)
        if os.path.isfile(tmp_dest_file_path):
            dest_file = os.path.join(combined_folder_code, dest_file_name)
            file_mapping[dest_file] = src_file
    current_idx += 1

# Next, copy the generated MILPs to the combined_folder_code
folder_paths_with_idx = []
for folder in os.listdir(gen_folder):
    folder_path = os.path.join(gen_folder, folder)
    if os.path.isdir(folder_path):
        idx, method = folder.split('_')
        folder_paths_with_idx.append((int(idx), folder_path))

# Sort folder paths based on indices
folder_paths_with_idx.sort(key=lambda x: x[0])

for idx, folder_path in folder_paths_with_idx:
    print('current idx', current_idx)
    if gen_start_idx <= idx < gen_end_idx:
        hard_file = os.path.join(folder_path, 'new_milp_hard.py')
        medium_file = os.path.join(folder_path, 'new_milp_medium.py')
        easy_file = os.path.join(folder_path, 'new_milp_easy.py')

        hard_log = os.path.join(folder_path, 'new_log_hard.txt')
        medium_log = os.path.join(folder_path, 'new_log_medium.txt')
        easy_log = os.path.join(folder_path, 'new_log_easy.txt')
        
        hard_solved = os.path.isfile(hard_file) and os.path.isfile(hard_log)
        medium_solved = os.path.isfile(medium_file) and os.path.isfile(medium_log)
        easy_solved = os.path.isfile(easy_file) and os.path.isfile(easy_log)

        chosen_hard_time = None
        chosen_medium_time = None
        chosen_curr = False
        # choose hard file first
        if hard_solved:
            print(f'{idx} use hard as hard')
            dest_file = os.path.join(combined_folder_code, f'milp_{current_idx}-hard.py')
            shutil.copy(hard_file, dest_file)
            shutil.copy(hard_log, os.path.join(combined_folder_log, f'milp_{current_idx}-hard.txt'))
            file_mapping[dest_file] = hard_file
            chosen_hard_time = extract_solve_time(hard_log)
            chosen_curr = True
            
        elif medium_solved:
            medium_time = extract_solve_time(medium_log)
            if medium_time and medium_time > HARD_TIME_TH:
                print(f'{idx} use medium as hard')
                dest_file = os.path.join(combined_folder_code, f'milp_{current_idx}-hard.py')
                shutil.copy(medium_file, dest_file)
                shutil.copy(medium_log, os.path.join(combined_folder_log, f'milp_{current_idx}-hard.txt'))
                file_mapping[dest_file] = medium_file
                chosen_hard_time = medium_time
                chosen_curr = True
                
        elif easy_solved:
            easy_time = extract_solve_time(easy_log)
            if easy_time and easy_time > HARD_TIME_TH:
                print(f'{idx} use easy as hard')
                dest_file = os.path.join(combined_folder_code, f'milp_{current_idx}-hard.py')
                shutil.copy(easy_file, dest_file)
                shutil.copy(easy_log, os.path.join(combined_folder_log, f'milp_{current_idx}-hard.txt'))
                file_mapping[dest_file] = easy_file
                chosen_hard_time = easy_time
                chosen_curr = True
                

        # choose medium file next
        if medium_solved:
            medium_time = extract_solve_time(medium_log)
            if medium_time:  # so the medium file is not chosen as the hard file
                if not (chosen_hard_time and medium_time > chosen_hard_time - TIME_GAP):
                    # if the medium file has a similar solve time as the chosen hard file, skip
                    print(f'{idx} use medium as medium')
                    dest_file = os.path.join(combined_folder_code, f'milp_{current_idx}-medium.py')
                    shutil.copy(medium_file, dest_file)
                    shutil.copy(medium_log, os.path.join(combined_folder_log, f'milp_{current_idx}-medium.txt'))
                    file_mapping[dest_file] = medium_file
                    chosen_medium_time = medium_time
                    chosen_curr = True
                    
        if not chosen_medium_time and easy_solved:
            easy_time = extract_solve_time(easy_log)
            if easy_time and easy_time > MEDIUM_TIME_TH: 
                if not (chosen_hard_time and easy_time > chosen_hard_time - TIME_GAP):
                    # if the easy file has a similar solve time as the chosen hard file, skip
                    print(f'{idx} use easy as medium')
                    dest_file = os.path.join(combined_folder_code, f'milp_{current_idx}-medium.py')
                    shutil.copy(easy_file, dest_file)
                    shutil.copy(easy_log, os.path.join(combined_folder_log, f'milp_{current_idx}-medium.txt'))
                    file_mapping[dest_file] = easy_file
                    chosen_medium_time = easy_time
                    print('save from', easy_file, ' to medium', dest_file)
                    chosen_curr = True 
                
        # choose easy file last
        if easy_solved:
            easy_time = extract_solve_time(easy_log)
            if easy_time:
                if not (chosen_hard_time and easy_time > chosen_hard_time - TIME_GAP) and \
                    not (chosen_medium_time and easy_time > chosen_medium_time - TIME_GAP): 
                    print(f'{idx} use easy as easy')

                    dest_file = os.path.join(combined_folder_code, f'milp_{current_idx}-easy.py')
                    shutil.copy(easy_file, dest_file)
                    shutil.copy(easy_log, os.path.join(combined_folder_log, f'milp_{current_idx}-easy.txt'))
                    file_mapping[dest_file] = easy_file
                    chosen_curr = True
    if chosen_curr:
        current_idx += 1

# Save the mapping to a JSON file
with open(os.path.join(combined_folder, 'file_mapping.json'), 'w') as f:
    json.dump(file_mapping, f, indent=4)
