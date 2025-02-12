import os
import pdb
import random
import itertools
import random
import re
import shutil
import json
import time
import numpy as np
from src.utils import do_oai_reply, get_parameters_str, edit_parameters, get_status_time_and_gap, parse_parameters_from_content, edit_parameters_from_content
from src.utils import run_and_save_log, check_feasibility, change_seed_and_check_feasibility, parse_parameters_from_content, multiprocess
from src.utils import parse_solving_process_from_content
from src.utils import parallel_fn


#### different seeds
DEFAULT_SEED = 42
TEST_SEEDS = [42]  # [1, 2, 3]
DEFAULT_GAP_TH = 0  # must solve to optimality at the given time limit
N_seeds = len(TEST_SEEDS)

GS_range_low = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
GS_range_high = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 35, 50]  # , 20, 50, 100
GS_range_both = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 35, 50]  # , 20, 50, 100
GS_range_mostly_low = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 3, 5, 10]  # , 20, 50, 100
GS_range_mostly_high = [0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 9, 15, 20, 35, 50]  # , 20, 50, 100
GS_range_mostly_high_short = [0.5, 0.75, 1, 2, 3, 5, 7, 9, 10, 15]

def convert_to_type(value):
    """Try to convert a string to int, float, or boolean."""
    if isinstance(value, list) or isinstance(value, tuple):
        return [convert_to_type(v) for v in value]
    if isinstance(value, str):
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value  # Return the original string if no conversion is possible


def generate_ranges_for_value(value, direction="inc"):
    """Generate valid ranges for a single value."""
    if isinstance(value, float) and 0 < value < 1:
        vmin, vmax = 0.1, 0.8
        if direction == 'inc':
            min_val, max_val = max(vmin, value), vmax
        elif direction == 'dec':
            min_val, max_val = vmin, min(vmax, value)
        else:
            min_val, max_val = vmin, vmax
        return [np.round(value, decimals=2) for value in np.linspace(min_val, max_val, 11).tolist()]
    elif isinstance(value, int) or isinstance(value, float):
        if direction == "inc":
            factor_range = GS_range_high
        elif direction == "dec":
            factor_range = GS_range_low
        elif direction == "mostly_inc":
            factor_range = GS_range_mostly_high
        elif direction == "mostly_dec":
            factor_range = GS_range_mostly_low
        elif direction == "mostly_inc_short":
            factor_range = GS_range_mostly_high_short
        else:
            factor_range = GS_range_both  # GS_range_low[:-1] + GS_range_high
        
        vmin, vmax = 0, 3000
        if value >= vmax or value <= vmin:
            return [value]
        # filter out factor range that is out of bound
        factor_range = [factor for factor in factor_range if vmin <= value * factor <= vmax]
        ranges = [np.round(value * factor, decimals=2) if isinstance(value, float) else int(value * factor) for factor in factor_range]
        return sorted(ranges)

    elif isinstance(value, bool):
        return [value, not value]
    else:
        return [value]  # Assuming no changes for strings or other types


def generate_valid_ranges(parameters, direction="inc"):
    valid_ranges = {}
    for param, value in parameters.items():
        converted_value = convert_to_type(value)
        if isinstance(converted_value, list) or isinstance(converted_value, tuple):
            ranges = []
            for v in converted_value:
                ranges.append(generate_ranges_for_value(v, direction=direction))
            # combine zip
            valid_ranges[param] = sorted(list(zip(*ranges)))
        else:
            valid_ranges[param] = generate_ranges_for_value(converted_value, direction=direction)
    return valid_ranges


def extract_grid_search_options(output_content, use_grid_search_only_prompt=False):
    if use_grid_search_only_prompt:
        section_str = "proposal and reasoning for grid search options for each parameter"
    else:
        section_str = "proposal and reasoning for new parameters and the grid search options"

    # Define patterns for extracting parameter names and grid search options
    param_pattern = re.compile(r'parameter[\s\*]*:\s*`(\w+)`\s*')
    options_pattern = re.compile(r'proposed grid search options[\s\*]*\s*:\s*(.+)\s*', re.IGNORECASE)

    lines = output_content.split('\n')

    # Find the relevant section
    start_index = None
    for i, line in enumerate(lines):
        if section_str in line.lower():
            start_index = i
    
    end_index = len(lines)
    grid_search_options = {}

    # If the section is found, extract parameters and options
    if start_index is not None and end_index is not None:
        i = start_index
        while i < end_index:
            param_match = param_pattern.search(lines[i].lower())
            if param_match:
                param_name = param_match.group(1)
                # Look ahead for the options pattern in the next few lines
                for j in range(i + 1, min(i + 10, end_index)):
                    options_match = options_pattern.search(lines[j].lower())
                    if options_match:
                        options_str = options_match.group(1).strip()
                        # Remove any enclosing backticks
                        if options_str.startswith('`') and options_str.endswith('`'):
                            options_str = options_str[1:-1]
                        # Convert the string to a list
                        options_list = eval(options_str)
                        grid_search_options[param_name] = options_list
                        break
            i += 1

    return grid_search_options


def get_param_history_str(param_history, time_limit=300, n_feasible=5, n_infeasible=3, n_random=3):
    feasible_solutions = [(params, solve_time) for params, solve_time in param_history if solve_time is not None]
    infeasible_solutions = [(params, solve_time) for params, solve_time in param_history if solve_time is None]
    feasible_solutions_sorted = sorted(feasible_solutions, key=lambda x: x[1], reverse=True)
    top_feasible = feasible_solutions_sorted[:n_feasible]
    remaining_feasible = feasible_solutions_sorted[n_feasible:]
    random_feasible = random.sample(remaining_feasible, min(n_random, len(remaining_feasible)))
    random_infeasible = random.sample(infeasible_solutions, min(n_infeasible, len(infeasible_solutions)))

    param_history_str = ""

    param_history_str += "## Top Parameters with Feasible Solutions Within Time Limit (Longer Solve Time): \n"
    for i, (params, solve_time) in enumerate(top_feasible):
        param_history_str += f"# Solution {i + 1}:\n"
        param_history_str += "# parameters = {\n"
        for key, value in params.items():
            param_history_str += f"    '{key}': {repr(value)},\n"
        param_history_str += "}\n"
        param_history_str += f"# Solve Time: {solve_time} seconds\n\n"

    param_history_str += "## Random Other Parameters with Feasible Solutions Within Time Limit: \n"
    for i, (params, solve_time) in enumerate(random_feasible, start=len(top_feasible) + 1):
        param_history_str += f"# Solution {i + 1}:\n"
        param_history_str += "# parameters = {\n"
        for key, value in params.items():
            param_history_str += f"    '{key}': {repr(value)},\n"
        param_history_str += "}\n"
        param_history_str += f"# Solve Time: {solve_time} seconds\n\n"

    param_history_str += "## Random Other Parameters Failed to Find Feasible Solution Within Time Limit: \n"
    for i, (params, solve_time) in enumerate(random_infeasible, start=len(top_feasible) + len(random_feasible) + 1):
        param_history_str += f"# Solution {i + 1}:\n"
        param_history_str += "# parameters = {\n"
        for key, value in params.items():
            param_history_str += f"    '{key}': {repr(value)},\n"
        param_history_str += "}\n"
        param_history_str += f"# Solve Time: N/A (Failed to find feasible solution within the time limit of {time_limit}s)\n\n"

    return param_history_str


def get_aggr_stats(i_stats_start, sample_result_list, feasibility_frac=1.0):
    # accumulated statistics for multiple seeds
    # sample_feasible, sample_criteria_met, sample_content, sample_log, sample_status, sample_solve_time, sample_gap
    if len(sample_result_list) == 0:
        return False, False, None, None
    aggr_feasible = sum([sample_result_list[i_stats_start+i_seed][0] for i_seed in range(N_seeds)]) >= feasibility_frac * N_seeds
    aggr_crteria_met = all([sample_result_list[i_stats_start+i_seed][1] for i_seed in range(N_seeds)]) if aggr_feasible else False
    aggr_solve_time = np.mean([sample_result_list[i_stats_start+i_seed][-2] for i_seed in range(N_seeds)]) if aggr_feasible else None
    aggr_gap = np.mean([sample_result_list[i_stats_start+i_seed][-1] for i_seed in range(N_seeds)]) if aggr_feasible else None
    
    return aggr_feasible, aggr_crteria_met, aggr_solve_time, aggr_gap


################# Filter proposed parameter based on a predefined set of criteria #################
def detailed_parse_log(log_data, file_path=None, 
                       solve_time_lb=20, solve_time_ub=400, 
                       presolve_time_proportion_ub=0.2, 
                       presolve_time_abs_ub=15.0, gap_ub=DEFAULT_GAP_TH, 
                       total_vars_lb=50, total_vars_ub=5e4, 
                       total_binint_vars_lb=50, total_binint_vars_ub=2e4, 
                       total_cons_lb=50, total_cons_ub=5e4, 
                       total_cons_types_lb=0, 
                       num_bnb_logs_lb=7, 
                       num_bnb_nodes_lb=10, num_bnb_nodes_ub=5e3,  # so that the task is more likely to be a valid task for branching
                       root_node_time_proportion_ub=0.3,  # may revisit this number  #  root_node_time_abs_ub=20.0,  # may revisit this number
                       integrality_gap_ub=3,  # aggressive threshold out large integrality gap 
                       detailed_logs=False,
                       verbose=True):
    # Parse the solving process log to extract stats
    if log_data.strip() == "":
        return False
    
    stats = parse_solving_process_from_content(log_data, detailed_logs=detailed_logs)
    
    # Unpack relevant stats
    total_solving_time = float('inf') if ("Total Solving Time" not in stats or stats["Total Solving Time"] is None) else stats["Total Solving Time"]
    total_presolving_time = float('inf') if ("Total Presolving Time" not in stats or stats["Total Presolving Time"] is None) else stats["Total Presolving Time"]
    gap = float('inf') if ("Gap" not in stats or stats["Gap"] is None) else stats["Gap"]
    total_vars = 0 if ("Total Variables" not in stats or stats["Total Variables"] is None) else stats["Total Variables"]
    binary_vars = 0 if ("Binary Variables" not in stats or stats["Binary Variables"] is None) else stats["Binary Variables"]
    integer_vars = 0 if ("Integer Variables" not in stats or stats["Integer Variables"] is None) else stats["Integer Variables"]
    total_cons = 0 if ("Total Constraints" not in stats or stats["Total Constraints"] is None) else stats["Total Constraints"]
    constraint_types = {} if ("Constraint Types" not in stats or stats["Constraint Types"] is None) else stats["Constraint Types"]
    num_bnb_logs = 0 if ("Number of BnB Logs" not in stats or stats["Number of BnB Logs"] is None) else stats["Number of BnB Logs"]
    num_bnb_nodes = 0 if ("Number of BnB Nodes" not in stats or stats["Number of BnB Nodes"] is None) else stats["Number of BnB Nodes"]
    integrality_gap = float('inf') if ("Integrality Gap" not in stats or stats["Integrality Gap"] is None) else stats["Integrality Gap"]
    root_time = total_solving_time if ("Root Time" not in stats or stats["Root Time"] is None) else stats["Root Time"]
    root_time_proportion = 1 if (root_time == float('inf') or total_solving_time == float('inf')) else root_time / total_solving_time if total_solving_time > 0 else 0

    presolve_time_proportion = 1 if total_presolving_time == float('inf') else total_presolving_time / total_solving_time if total_solving_time > 0 else 0
    total_binint_vars = binary_vars + integer_vars

    if stats["Gap"] == None:
        print('??????????????? why is gap none??????????????')
        print(log_data)
        print('?????????????????????????????????????')
        return 

    # Check criteria
    criteria_met = True
    output_message = "=========================================================================================\n"
    output_message += (f"[Orig file {file_path}] " if file_path else "") + f"Stats:: {stats}\n"


    if not (solve_time_lb <= total_solving_time <= solve_time_ub):
        criteria_met = False
        output_message += f"Solve time {total_solving_time} is outside bounds ({solve_time_lb}, {solve_time_ub}).\n"
    
    if not (presolve_time_proportion <= presolve_time_proportion_ub):
        criteria_met = False
        output_message += f"Presolve time proportion {presolve_time_proportion:.2f} exceeds upper bound {presolve_time_proportion_ub:.2f}.\n"
    
    if not (total_presolving_time <= presolve_time_abs_ub):
        criteria_met = False
        output_message += f"Total presolve time {total_presolving_time:.2f} exceeds upper bound {presolve_time_abs_ub:.2f}.\n"
    
    if not (gap <= gap_ub * 100):
        criteria_met = False
        output_message += f"Gap {gap:.2f}% exceeds upper bound {gap_ub*100:.2f}%.\n"
    
    if not (total_vars_lb <= total_vars <= total_vars_ub):
        criteria_met = False
        output_message += f"Total variables {total_vars} is outside bounds ({total_vars_lb}, {total_vars_ub}).\n"
    
    if not (total_binint_vars_lb <= total_binint_vars <= total_binint_vars_ub):
        criteria_met = False
        output_message += f"Total binary and integer variables {total_binint_vars} is outside bounds ({total_binint_vars_lb}, {total_binint_vars_ub}).\n"

    if not (total_cons_lb <= total_cons <= total_cons_ub):
        criteria_met = False
        output_message += f"Total constraints {total_cons} is outside bounds ({total_cons_lb}, {total_cons_ub}).\n"
    
    if not (len(constraint_types) >= total_cons_types_lb):
        criteria_met = False
        output_message += f"Number of constraint types {len(constraint_types)} is less than lower bound {total_cons_types_lb}.\n"
    
    if not (num_bnb_logs >= num_bnb_logs_lb):
        criteria_met = False
        output_message += f"Number of BnB logs {num_bnb_logs} is less than lower bound {num_bnb_logs_lb}.\n"

    if not num_bnb_nodes <= num_bnb_nodes_ub:  # (num_bnb_nodes_lb <= num_bnb_nodes <= num_bnb_nodes_ub):
        criteria_met = False
        output_message += f"Number of BnB nodes {num_bnb_nodes} is outside bound ({num_bnb_nodes_lb}, {num_bnb_nodes_ub}).\n"

    if not (integrality_gap <= integrality_gap_ub):
        criteria_met = False
        output_message += f"Integrality gap {integrality_gap} exceeds upper bound {integrality_gap_ub}.\n"

    output_message += "=========================================================================================\n"

    if not criteria_met and verbose:
        print(output_message.strip())
    
    return criteria_met


def evaluate_params_i(file_path, sample_param, i_sample, timeout_duration, subprocess_timeout_duration, 
                      seed=DEFAULT_SEED, gap_th=DEFAULT_GAP_TH, remove_tmp=True, save_tmp_file_path=None):
    new_content = edit_parameters(file_path, sample_param)

    if save_tmp_file_path:
        new_content_tmp_file = os.path.join(save_tmp_file_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_tmp_{i_sample}_{seed}.py")
    else:
        new_content_tmp_file = f"{os.path.splitext(file_path)[0]}_tmp_{i_sample}_{seed}.py"

    # write new_content to a tmp file
    with open(new_content_tmp_file, 'w') as f:
        f.write(new_content)
    
    # run the script and get log (can change seed here as well)
    if seed != DEFAULT_SEED:
        new_feasible, _, new_log = change_seed_and_check_feasibility(new_content_tmp_file, seed, 
            timeout_duration=timeout_duration, subprocess_timeout_duration=subprocess_timeout_duration, gap_th=gap_th)
    else:
        new_feasible, _, new_log = check_feasibility(new_content_tmp_file, 
            timeout_duration=timeout_duration, subprocess_timeout_duration=subprocess_timeout_duration, gap_th=gap_th)

    new_status, new_solve_time, new_gap = get_status_time_and_gap(new_log)

    new_criteria_met = detailed_parse_log(new_log, file_path) if new_feasible else False
    print(f'Param {i_sample} ({new_content_tmp_file})::', 'Feasible.' if new_feasible else 'Infeasible.', f'New status = {new_status}, solve time = {new_solve_time}s, gap = {new_gap}, criteria_met = {new_criteria_met}.')

    # new_content_tmp_log = f"{os.path.splitext(file_path)[0]}_tmp_{i_sample}_{seed}_log.txt"
    # open(new_content_tmp_log, "w").write(new_log)
    # open(new_content_tmp_file, "w").write(new_content)
    # remove the tmp file
    if remove_tmp:
        if os.path.exists(new_content_tmp_file):
            os.remove(new_content_tmp_file)
    else:
        new_log_tmp_file = f"{os.path.splitext(file_path)[0]}_tmp_{i_sample}_{seed}_log.txt"
        with open(new_log_tmp_file, 'w') as f:
            f.write(new_log)
        if not os.path.exists(new_content_tmp_file):
            with open(new_content_tmp_file, 'w') as f:
                f.write(new_content)

    return new_feasible, new_criteria_met, new_content, new_log, new_status, new_solve_time, new_gap



def evaluate_params_i_args(args):
    file_path, sample_param, i_sample, timeout_duration, subprocess_timeout_duration, seed, remove_tmp, save_tmp_file_path = args
    if i_sample % 10 == 0:
        print(f'Start evaluate {i_sample} with seed {seed}...')
    return evaluate_params_i(file_path, sample_param, i_sample, timeout_duration, subprocess_timeout_duration, 
                             seed=seed, remove_tmp=remove_tmp, save_tmp_file_path=save_tmp_file_path)
   

def grid_search_from_options(parameters, grid_search_options, n_samples=10, shuffle=True, do_sample=True,
                             sample_max_time=60):
    def make_hashable(item):
        if isinstance(item, (list, tuple)):
            return tuple(make_hashable(sub_item) for sub_item in item)
        elif isinstance(item, dict):
            return tuple(sorted((key, make_hashable(value)) for key, value in item.items()))
        elif isinstance(item, set):
            return frozenset(make_hashable(sub_item) for sub_item in item)
        else:
            return item
    
    # Function to check if min and max constraints are satisfied
    def is_feasible(combination):
        for name, value in combination.items():
            if name.startswith("min_") and f"max_{name[4:]}" in combination:
                if value > combination[f"max_{name[4:]}"]:
                    return False
            if name.endswith("_lower_bound") and f"{name[:-12]}_upper_bound" in combination:
                if value > combination[f"{name[:-12]}_upper_bound"]:
                    return False
        return True
    
    # Define parameter names and calculate the number of possible combinations
    param_names = list(parameters.keys())
    
    n_combinations, cap = 1, 1e6
    for param in param_names:
        n_combinations *= len(grid_search_options[param])
        if n_combinations >= cap or n_combinations < 0:
            n_combinations = cap
            break

    n_samples = min(n_combinations, n_samples)
    
    # Generate grid search samples
    if do_sample:
        attempts = 0
        max_attempts = 1e5  # Prevent sampling too many
        unique_samples = set()
        grid_search_samples = []
        start_time = time.time()
        while attempts < max_attempts and len(grid_search_samples) < n_samples and len(unique_samples) < n_combinations:
            sample = tuple(random.choice(grid_search_options[param]) for param in param_names)
            hashable_sample = make_hashable(sample)
            attempts += 1
            
            if hashable_sample not in unique_samples:
                try:
                    unique_samples.add(hashable_sample)
                except:
                    print(f"Sample {sample} is not hashable!")
                    # write the sample to a json file
                    with open('unhashable_sample.json', 'w') as f:
                        json.dump(sample, f)

                sample_dict = dict(zip(param_names, sample))
                if is_feasible(sample_dict):
                    grid_search_samples.append(sample)
            if time.time() - start_time > sample_max_time:
                print(f'Time limit reached for sampling. {len(grid_search_options)} samples generated.')
                print('Grid search options:', grid_search_options)
                break
        # print('# attempts', attempts, 'n_samples', n_samples, 'n_combinations', n_combinations, 'len(unique_feasible_samples)', len(unique_feasible_samples), 'len(unique_samples)', len(unique_samples))
    else:
        # Generate all combinations and filter feasible ones
        param_combinations = list(itertools.product(*[grid_search_options[param] for param in param_names]))
        feasible_combinations = [combo for combo in param_combinations if is_feasible(dict(zip(param_names, combo)))]
        grid_search_samples = feasible_combinations[:n_samples]

    grid_search_samples = [dict(zip(param_names, combination)) for combination in grid_search_samples]
    if shuffle:
        random.shuffle(grid_search_samples)

    return grid_search_samples


def grid_search_from_range(parameters, valid_ranges, n_samples=10, shuffle=True):
    """Generate parameter combinations for grid search."""
    param_names = parameters.keys()
    grid_search_options = {param: valid_ranges[param] for param in param_names}

    return grid_search_from_options(parameters, grid_search_options, n_samples=n_samples, shuffle=shuffle)


def prompt_search(file_path, log_path, model_name='gpt-4o'):
    # prompt gpt to adjust the parameters
    milp = open(file_path, "r").read()
    cur_params = get_parameters_str(file_path)
    log = open(log_path, "r").read()
    solve_time = get_status_time_and_gap(log)[1]
    system_message = SYSTEM_MESSAGE_PARAM

    prompt = PROMPT_PARAM.format(solve_time=solve_time, milp=milp, cur_params=cur_params, log=log)
    print(prompt)

    print(f'=================================== Start prompt ===================================')
    answers_i = do_oai_reply(system_message, prompt, model_name=model_name)
    # answers.append(answers_i[0] if len(answers_i) > 0 else "")
    return answers_i


def save_param_history(save_dir_i, file_path, param_history, new_params, max_solve_time_within_range, 
                       timeout_duration=300, subprocess_timeout_duration=600, final_time_th=5):
    # maintain all the parameters, save the one with 1/3, 2/3, 3/3 solve time
    param_history_filter_sort = sorted([hist for hist in param_history if hist[1] > final_time_th], key=lambda x: x[1])
    print('---------------------------- Save Param History Statistics ----------------------------')
    print(f'Min solve time = {min([hist[1] for hist in param_history]):.2f}s, max solve time = {max([hist[1] for hist in param_history]):.2f}s, '
          f'median solve time = {np.median([hist[1] for hist in param_history]):.2f}s, mean solve time = {np.mean([hist[1] for hist in param_history]):.2f}s, '
          f'std = {np.std([hist[1] for hist in param_history]):.2f}s. Max Solve Time Within Range = {max_solve_time_within_range:.2f}s.')
    
    if len(param_history_filter_sort) > 0:
        unique_param_idx = set()
        for frac_diff, difficulty_level in [(1, 'hard'), (2.0 / 3, 'medium'), (1.0 / 3, 'easy')]:
            threshold_time = frac_diff * timeout_duration
            param_idx = np.searchsorted([hist[1] for hist in param_history_filter_sort], threshold_time, side='left')                
            if param_idx not in unique_param_idx and 0 <= param_idx < len(param_history_filter_sort):
                unique_param_idx.add(param_idx)
                new_milp_path_diff = os.path.join(save_dir_i, f"new_milp_{difficulty_level}.py") 
                new_log_path_diff = os.path.join(save_dir_i, f"new_log_{difficulty_level}.txt")
                new_params_diff, new_solve_time_diff = param_history_filter_sort[param_idx]
                new_milp_diff = edit_parameters(file_path, new_params_diff)
                open(new_milp_path_diff, "w").write(new_milp_diff)
                success_diff = run_and_save_log(new_milp_path_diff, new_log_path_diff, timeout_duration=timeout_duration, subprocess_timeout_duration=subprocess_timeout_duration)
                if success_diff:
                    new_log_diff = open(new_log_path_diff, "r").read()
                    new_status_diff, new_solve_time_diff, new_gap_diff = get_status_time_and_gap(new_log_diff)
                    print(f'Final difficulty level {difficulty_level}:: selection: params = ', new_params_diff, f'status = {new_status_diff}, solve time = {new_solve_time_diff}s, gap = {new_gap_diff}.')
                else:
                    print(f'Final difficulty level {difficulty_level}:: failed')

    success = False
    if new_params is not None:
        # save to new milp
        new_milp_path = os.path.join(save_dir_i, f"new_milp.py")
        new_log_path = os.path.join(save_dir_i, f"new_log.txt")
        new_milp = edit_parameters(file_path, new_params)
        open(new_milp_path, "w").write(new_milp)
        success = run_and_save_log(new_milp_path, new_log_path, timeout_duration=timeout_duration, subprocess_timeout_duration=subprocess_timeout_duration)
        if success:
            new_log = open(new_log_path, "r").read()
            new_status, new_solve_time, new_gap = get_status_time_and_gap(new_log)
            print(f'Final selection: params = ', new_params, f'New status = {new_status}, solve time = {new_solve_time}s, gap = {new_gap}.')
            if new_solve_time < final_time_th:
                print('Final solution is too short! Remove the instance')
                success = False
    return success


def append_idx_to_json(json_path, idx):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            idxs = json.load(f)
        idxs.append(idx)
    else:
        idxs = [idx]
    with open(json_path, 'w') as f:
        json.dump(idxs, f)


def adjust_params_heuristics_search_i(org_params, org_solve_time, save_file_path, save_dir_i, 
                                      n_samples_per_idx=10, timeout_duration=300, subprocess_timeout_duration=600,
                                      range_direction='inc', inc_th=30, n_round=1, n_cpus=None, final_time_th=5,
                                      remove_tmp=True, save_tmp_file_path=None, save_satisfy=False, save_satisfy_file_path=None):
    print('------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------')
    print(f'Start adjust_params_heuristics_search_i for {save_dir_i}...')
    ############################ Heuristic search for the best parameter ############################
    param_history = [(org_params, org_solve_time)]

    new_params = None
    max_solve_time_within_range = -float('inf')   # (0, timeout_duration)
    for i_round in range(n_round):
        new_milp_path_i = os.path.join(save_dir_i, f"new_milp_{i_round}.py")
        new_log_path_i = os.path.join(save_dir_i, f"new_log_{i_round}.txt")

        parameters = org_params if new_params is None else new_params 
        if max(org_solve_time, max_solve_time_within_range) <= inc_th:
            range_direction = "inc"         # if solve time is less than inc_th, then must increase the range
        elif max(org_solve_time, max_solve_time_within_range) > timeout_duration - inc_th:
            range_direction = "mostly_dec"         # if solve time is already pretty high, then mostly decrease the range
        valid_ranges = generate_valid_ranges(parameters, direction=range_direction)
        samples = grid_search_from_range(parameters, valid_ranges, n_samples=n_samples_per_idx)
        
        tasks = [(save_file_path, sample_param, i_sample, timeout_duration, subprocess_timeout_duration, 
                  seed, remove_tmp, save_tmp_file_path) for i_sample, sample_param in enumerate(samples) for seed in TEST_SEEDS]
        
        sample_result_list = parallel_fn(evaluate_params_i_args, tasks, n_cpus=n_cpus)
        aggregated_results = [(i_sample, sample_param, *get_aggr_stats(i_sample*N_seeds, sample_result_list)) for i_sample, sample_param in enumerate(samples)]
        
        # update new parameter if it has a longer solve time
        new_params_updated = False
        i_satisfy = 0
        for i_sample, sample_param, sample_feasible, sample_criteria_met, sample_solve_time, sample_gap in aggregated_results:
            if sample_feasible and sample_criteria_met:
                param_history.append((sample_param, sample_solve_time))
                if sample_solve_time > max_solve_time_within_range:
                    max_solve_time_within_range = sample_solve_time
                    new_params = sample_param
                    new_params_updated = True
                
                if save_satisfy and save_satisfy_file_path and os.path.exists(save_satisfy_file_path):
                    # write the code and the log to file
                    curr_milp = sample_result_list[i_sample][2]
                    curr_log = sample_result_list[i_sample][3]
                    curr_milp_path = os.path.join(save_satisfy_file_path, f"milp_{i_round}_{i_satisfy}.py")
                    curr_log_path = os.path.join(save_satisfy_file_path, f"log_{i_round}_{i_satisfy}.txt")
                    open(curr_milp_path, "w").write(curr_milp)
                    open(curr_log_path, "w").write(curr_log)
                    i_satisfy += 1
        
        del sample_result_list  # free memory 

        print(f'Heuristic search round {i_round}: #{len(samples)} parameter samples with {N_seeds} seeds, '
              f'max solve time so far {max([solve_time for params, solve_time in param_history if solve_time is not None], default=None)}s.')              

        if new_params_updated:
            # save to new milp
            new_milp = edit_parameters(save_file_path, new_params)
            open(new_milp_path_i, "w").write(new_milp)
            # print(f'Run and check the new milp under new param {new_params}...')
            success = run_and_save_log(new_milp_path_i, new_log_path_i, timeout_duration=timeout_duration, subprocess_timeout_duration=subprocess_timeout_duration, remove_tmp=remove_tmp)
            if success:
                new_status, new_solve_time, new_gap = get_status_time_and_gap(open(new_log_path_i).read())
                print(f'!!!!!!!!!!!!!Round {i_round}:: New status = {new_status}, solve time = {new_solve_time}s, gap = {new_gap}!!!!!!!!!!!!!!!!!!')
            else:
                print('!!!!!!!!!!!!!!Round 1 failed to find feasible solution within the time limit!!!!!!!!!!!!!!!!!')
                if remove_tmp:
                    for new_file_path_i in [new_milp_path_i, new_log_path_i]:
                        if os.path.exists(new_file_path_i):
                            os.remove(new_file_path_i)
    #################################################################################################

    ################################# Save the generated parameters #################################
    success = save_param_history(save_dir_i, save_file_path, param_history, new_params, max_solve_time_within_range, timeout_duration=timeout_duration, 
                                 subprocess_timeout_duration=subprocess_timeout_duration, final_time_th=final_time_th)
    #################################################################################################
    
    success_str = 'Success' if success else 'Failed'
    print(f'End adjust_params_heuristics_search_i for {save_dir_i}... {success_str}')

    return success


def adjust_params_heuristics_search(milp_and_log_paths, save_dir, n_samples_per_idx=10, timeout_duration=300, subprocess_timeout_duration=600,
                                    range_direction='inc', n_round=1, n_cpus=None, final_time_th=5, save_satisfy=False):
    if len(milp_and_log_paths) == 0:
        print("No milps to generate prompts from.")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating Parameters to {save_dir}...")
    forbidden_json_path = os.path.join(save_dir, 'forbidden.json')
    generated_json_path = os.path.join(save_dir, 'generated.json')

    for idx in range(len(milp_and_log_paths)):
        print('-----------------------------------------------------------------------')
        # path of the original milp code and log
        org_file_path, log_path = milp_and_log_paths[idx]
        ########################## path of the original milp code and log ###############################
        milp_code = open(org_file_path, "r").read()

        try:
            org_params = parse_parameters_from_content(milp_code)
        except:
            print(f"Error: Cannot parse parameters from {org_file_path}.")
            continue

        log = open(log_path, "r").read()
        status, solve_time, gap = get_status_time_and_gap(log)
        match = re.search(r"\/(\d+)_.*\/milp.py", org_file_path)
        if match:
            org_idx = str(match.group(1))
        else:
            org_idx = os.path.basename(org_file_path).split('.')[0].split('_')[-1]
                   
        print(f"Index {org_idx} grid search for file path {org_file_path}...")
        print('Original parameter:', org_params)
        print(f'Original status = {status}, solve time = {solve_time}s, gap = {gap}. ')
        #################################################################################################

        #################################### new saved result directory #################################
        parent_save_dir_i = os.path.join(save_dir, f"result_{org_idx}")
        # remove save_dir_i if it exists
        if os.path.exists(parent_save_dir_i):
            shutil.rmtree(parent_save_dir_i)
            
        os.makedirs(parent_save_dir_i, exist_ok=True)

        save_dir_i = os.path.join(parent_save_dir_i, 'param_search_dir')
        os.makedirs(save_dir_i, exist_ok=True)
        
        save_file_path = os.path.join(save_dir_i, f"milp.py")
        save_log_path = os.path.join(save_dir_i, f"log.txt")
        open(save_file_path, "w").write(milp_code)
        open(save_log_path, "w").write(log)

        if save_satisfy:
            save_satisfy_file_path = os.path.join(parent_save_dir_i, 'param_search_satisfy_dir')
            if os.path.exists(save_satisfy_file_path):
                shutil.rmtree(save_satisfy_file_path)
            os.makedirs(save_satisfy_file_path, exist_ok=True)

        #################################################################################################
        success = adjust_params_heuristics_search_i(org_params, solve_time, save_file_path, save_dir_i, 
                                                    n_samples_per_idx=n_samples_per_idx, timeout_duration=timeout_duration, 
                                                    subprocess_timeout_duration=subprocess_timeout_duration,
                                                    range_direction=range_direction, n_round=n_round, 
                                                    n_cpus=n_cpus, final_time_th=final_time_th,
                                                    save_satisfy=save_satisfy, save_satisfy_file_path=save_satisfy_file_path)
           
        ### update the generated and forbidden json files
        if success:
            append_idx_to_json(generated_json_path, org_idx)
        else:
            print(f'No new parameter found for instance index {org_idx}::: Remove the instance index...')
            shutil.rmtree(save_dir_i)
            append_idx_to_json(forbidden_json_path, org_idx)
        #################################################################################################