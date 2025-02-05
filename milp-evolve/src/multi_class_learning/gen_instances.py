import os
import glob
import sys
import numpy as np
import json
import argparse
import shutil
from utils import parallel_fn, change_seed_and_check_feasibility, get_status_time_and_gap
import gzip
from utils import get_compress_path, get_files_by_extension
import time
from pyscipopt import Model
import multiprocessing as mp
import re
import subprocess


############################# Helper functions #############################
def change_seed_in_file(file_path, new_seed, tmp_file_path=None, mps_file=None):
    with open(file_path, 'r') as original_file:
        lines = original_file.readlines()

    seed_pattern = re.compile(r'(seed\s*=\s*)\d+')
    objective_pattern_start = re.compile(r'(\s*)model\.setObjective\(\s*')
    objective_pattern_maximize = re.compile(r'\s*["\']maximize["\']\s*')
    objective_pattern_end = re.compile(r'\s*\)\s*')

    if tmp_file_path is None:
        tmp_file_path = f"{os.path.splitext(file_path)[0]}_seed{new_seed}.py"

    if '"maximize"' in ''.join(lines) or '\'maximize\'' in ''.join(lines):
        objective_found = False
        buffer = []
        inside_objective = False
        maximize_found = False
        seed_found = False

        with open(tmp_file_path, 'w') as file:
            for line in lines:
                # Update the seed
                if not seed_found and re.search(seed_pattern, line):
                    seed_found = True
                    line = re.sub(seed_pattern, r'\g<1>' + str(new_seed), line)
                    file.write(line)
                    continue

                # Check for the start of the objective
                if not objective_found and re.search(objective_pattern_start, line):
                    inside_objective = True

                if inside_objective:
                    buffer.append(line)

                    # Check for maximize within the objective
                    if re.search(objective_pattern_maximize, line):
                        line = re.sub(objective_pattern_maximize, lambda match: match.group(0).replace('\'maximize\'', '"minimize"').replace('"maximize"', '"minimize"'), line)
                        buffer[-1] = line
                        maximize_found = True

                    # Check for the end of the objective only if maximize was found
                    if maximize_found and re.search(objective_pattern_end, line):
                        objective_found, inside_objective, maximize_found = True, False, False
                        full_objective = ''.join(buffer)
                        new_objective_expr = full_objective.replace('model.setObjective(', 'model.setObjective(-(')
                        # new_objective_expr = re.sub(r',\s*"minimize"', '), "minimize"', new_objective_expr)
                        # new_objective_expr = re.sub(r",\s*'minimize'", "), 'minimize'", new_objective_expr)
                        new_objective_expr = re.sub(r'(,\s*\n*\s*)"minimize"', r')\1"minimize"', new_objective_expr)
                        new_objective_expr = re.sub(r"(,\s*\n*\s*)'minimize'", r")\1'minimize'", new_objective_expr)
                        file.write(new_objective_expr)
                        buffer = []
                    continue

                # Write to the temporary file
                if mps_file is not None and 'model.optimize()' in line:
                    indentation = len(line) - len(line.lstrip())
                    problem_line = ' ' * indentation + f"model.writeProblem('{mps_file}')\n"
                    file.write(problem_line)
                    continue

                file.write(line)
                    
        # Save to debug_mps.py if the objective line is not found
        if not objective_found:
            print('Maximize: Objective line not found in', file_path)
            with open(tmp_file_path, 'w') as file:
                file.writelines(lines)

            debug_file_path = "debug_mps.py"
            with open(debug_file_path, 'w') as debug_file:
                debug_file.writelines(lines)

    else:
        seed_found = False

        with open(tmp_file_path, 'w') as file:
            for line in lines:
                # Update the seed
                if not seed_found and re.search(seed_pattern, line):
                    seed_found = True
                    line = re.sub(seed_pattern, r'\g<1>' + str(new_seed), line)
                    file.write(line)
                    continue

                # Write to the temporary file
                if mps_file is not None and 'model.optimize()' in line:
                    indentation = len(line) - len(line.lstrip())
                    problem_line = ' ' * indentation + f"model.writeProblem('{mps_file}')\n"
                    file.write(problem_line)
                    continue

                file.write(line)

    # Verify file is not empty or just space
    with open(tmp_file_path, 'r') as file:
        if not any(line.strip() for line in file):
            print('Empty file', tmp_file_path)

            with open(file_path, 'r') as new_file:
                if not any(line.strip() for line in new_file):
                    print('Empty original file', file_path)

    return tmp_file_path


def check_valid_file(script_path, timeout_duration=20):
    try:
        result = subprocess.run(
            ['python', script_path], 
            capture_output=True, 
            text=True, 
            timeout=timeout_duration
        )
        
        # Check if there was any error
        if result.returncode != 0:
            # print(result.stderr.strip())
            return False  # f"Script has bugs. Error: {result.stderr.strip()}"
        else:
            return True  # "Script ran successfully. No bugs detected."
    
    except subprocess.TimeoutExpired:
        return True  # "Script timed out. It might be stuck or have performance issues."
    except Exception as e:
        return False  # f"An error occurred while running the script: {str(e)}"


def change_seed_and_check_valid_file(file_path, new_seed, timeout_duration=20,
                                     mps_file=None, remove_tmp=True, tmp_file_path=None):
    tmp_file_path = change_seed_in_file(file_path, new_seed, mps_file=mps_file, tmp_file_path=tmp_file_path)
    valid = check_valid_file(tmp_file_path, timeout_duration=timeout_duration)
    # remove tmp file
    if remove_tmp and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    return valid

############################# MILP code data generation #############################
def generate_instance_seed_with_args(args):
    milp_filename, instances_dir_p, info_dir_p, log_dir_p, seed, optimize, gap_th, \
        valid_timeout_duration, timeout_duration, subprocess_timeout_duration, remove_tmp = args
    
    ########################## First generate mps files without gzip compression ############################
    instance_mps = os.path.join(instances_dir_p, f'{seed}.mps')
    log_file = os.path.join(log_dir_p, f'{seed}.txt')
    tmp_file_path = os.path.join(instances_dir_p, f'seed{seed}.py')

    if optimize:
        print(f'Generate instance {instance_mps} with seed {seed}')
        feasible, solve_time, output = change_seed_and_check_feasibility(milp_filename, seed, timeout_duration=timeout_duration, 
                                                                        subprocess_timeout_duration=subprocess_timeout_duration, gap_th=gap_th, 
                                                                        tmp_file_path=tmp_file_path, mps_file=instance_mps, remove_tmp=remove_tmp)
        new_kwargs = {'code': milp_filename, 'seed': seed, 'feasible': feasible, 'solve_time': solve_time}
        status, solve_time, gap = get_status_time_and_gap(output)
        if not feasible or status not in ['optimal', 'feasible', 'timelimit'] or gap is None or gap > gap_th:
            # not feasible! remove this instance: remove milp_filename, remove log_file, remove tmp_file_path if an exsits
            print(f'Not feasible or gap = {gap} is too big! Remove this instance {instance_mps}.')
            for path in [instance_mps, log_file, tmp_file_path]:
                if os.path.exists(path):
                    os.remove(path)
            return False
            
        open(log_file, "w").write(output)
    else:
        print(f'Generate {milp_filename} - seed {seed} - timeout duration {valid_timeout_duration} - tmp file path {tmp_file_path} - mps file {instance_mps} - remove tmp {remove_tmp}')
        valid = change_seed_and_check_valid_file(milp_filename, seed, timeout_duration=valid_timeout_duration,                                                     
                                                 tmp_file_path=tmp_file_path, mps_file=instance_mps, remove_tmp=remove_tmp)
        if not valid:
            print(f'Invalid! Remove this instance {milp_filename}.')
            for path in [instance_mps, log_file, tmp_file_path]:
                if os.path.exists(path):
                    os.remove(path)
            return False

        new_kwargs = {'code': milp_filename, 'seed': seed, 'valid': valid}

    ########################## Then compress the mps files ############################
    if os.path.exists(instance_mps):
        ## Try to read the mps file: there may be error reading it. If error occurs, return failure as well
        try:
            model = Model()
            model.readProblem(instance_mps)
        except Exception as e:
            print(f'Error reading the MPS file {instance_mps}: {e}')
            for path in [instance_mps, log_file, tmp_file_path]:
                if os.path.exists(path):
                    os.remove(path)
            return False
    
        with open(instance_mps, 'rb') as f_in, gzip.open(get_compress_path(instance_mps), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(instance_mps)
    
    json_info = os.path.join(info_dir_p, f'seed{seed}.json')
    json_object = json.dumps(new_kwargs)
    open(json_info, "w").write(json_object)
    print(f'End of generating instance {instance_mps} with seed {seed}' + (f' and save the log to {log_file}' if optimize else ''))
    return True


def worker(task_queue, results):
    while True:
        helper_args = task_queue.get()
        if helper_args is None:
            break
        milp_filename = helper_args[0]

        try:
            success = generate_instance_seed_with_args(helper_args)
        except Exception as e:
            print(f"MILP file {milp_filename} encountered an unexpected error: {e}")
            success = False
        
        results[milp_filename] = success


def run_multiprocess(helper_args_list, num_processes, func):
    with mp.Manager() as manager:
        task_queue = manager.Queue()
        results = manager.dict()
        # Populate the task queue
        for args in helper_args_list:
            task_queue.put(args)  # last two are label saves

        # Signal end of tasks
        for _ in range(num_processes):
            task_queue.put(None)

        processes = [mp.Process(target=worker, args=(task_queue, results)) for _ in range(num_processes)]
        
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return [results.get(helper_args[0], False) for helper_args in helper_args_list]


def generate_instances(milp_filename, instances_dir, n_instances=10, optimize=True, 
                       gap_th=0, valid_timeout_duration=10, timeout_duration=100, 
                       subprocess_timeout_duration=200, remove_tmp=True, n_cpus=None,
                       code_str='code'):
    instances_dir = os.path.join(instances_save_dir, 'mps')
    info_dir = os.path.join(instances_save_dir, 'info')
    log_dir = os.path.join(instances_save_dir, 'logs')
    os.makedirs(instances_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # change a few seeds to generate different instances
    basename_without_ext = os.path.basename(milp_filename).split('.')[0]
    instances_dir_p = os.path.join(instances_dir, code_str, basename_without_ext)
    info_dir_p = os.path.join(info_dir, code_str, basename_without_ext)
    log_dir_p = os.path.join(log_dir, code_str, basename_without_ext)
    os.makedirs(instances_dir_p, exist_ok=True)
    os.makedirs(info_dir_p, exist_ok=True)
    os.makedirs(log_dir_p, exist_ok=True)

    
    tasks = []
    instance_i = 0
    for seed in range(n_instances):
        instance_mps = os.path.join(instances_dir_p, f'{seed}.mps.gz')
        if any(os.path.exists(path) for path in get_files_by_extension(instances_dir_p, file_prefix=f'{seed}')): #  and (not optimize or os.path.exists(log_file)):
            print(f'Instance {instance_mps} already exists. Skip.')
            instance_i += 1
        else:
            tasks.append((milp_filename, instances_dir_p, info_dir_p, log_dir_p, seed, optimize, gap_th, 
            valid_timeout_duration, timeout_duration, subprocess_timeout_duration, remove_tmp))
            
    print(f'{len(tasks)} tasks to do out of {n_instances} instances.')

    success_results = run_multiprocess(tasks, n_cpus, generate_instance_seed_with_args)
    instance_i = 0

    for i_result, success in enumerate(success_results):
        idx = tasks[i_result][4]
        instance_mps = os.path.join(instances_dir_p, f'{idx}.mps')
        instance_mps_gz = os.path.join(instances_dir_p, f'{idx}.mps.gz')
        log_file = os.path.join(log_dir_p, f'{idx}.txt')
        info_file = os.path.join(info_dir_p, f'seed{idx}.json')
        if success:
            instance_i += 1
        else:
            # remove the instance
            for path in [instance_mps, instance_mps_gz, log_file, info_file]:
                if os.path.exists(path):
                    os.remove(path)

    
    print(f'Generated {instance_i} instances of python file {milp_filename} to {instances_dir_p} directory.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process configuration settings for data handling and processing.")

    parser.add_argument("--n_instances", type=int, default=1000, help="Number of samples.")
    parser.add_argument("--n_cpus", type=int, default=10, help="Number of CPUs to use.")
    parser.add_argument("--instances_save_dir", type=str, default='save_dir/instances', help="Directory for instance data.")
    
    parser.add_argument("--milp_type", type=str, default='code', help="Type of MILP instances to process.")
    parser.add_argument("--difficulty", type=str, default='all', help="Difficulty of instances to process.")
    parser.add_argument("--code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--code_dir", type=str, default='milp_code', help="Directory for MILP code files")
    parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")
    
    parser.add_argument("--valid_timeout_duration", type=int, default=30, help="Time limit for SCIP solver to check if the code is valid.")
    parser.add_argument("--timeout_duration", type=int, default=360, help="Time limit for SCIP solver.")
    parser.add_argument("--subprocess_timeout_duration", type=int, default=540, help="Time limit for subprocess.")
    parser.add_argument("--gap_th", type=float, default=0, help="Gap threshold at the timeout duration for the mps to be valid.")
    parser.add_argument("--not_optimize", action="store_true", help="Flag to optimize the instance.")
    parser.add_argument("--not_remove_tmp", action="store_true", help="Flag to remove temporary files.")
    args = parser.parse_args()


    instances_save_dir = args.instances_save_dir
    os.makedirs(instances_save_dir, exist_ok=True)

    n_instances = args.n_instances
    optimize = not args.not_optimize  # solve the instance and save the log
    remove_tmp = not args.not_remove_tmp
    print('optimize', optimize, 'remove_tmp', remove_tmp)

    n_cpus = args.n_cpus
    valid_timeout_duration = args.valid_timeout_duration
    timeout_duration = args.timeout_duration  # check valid time / scip solve time limit
    subprocess_timeout_duration = args.subprocess_timeout_duration

    difficulty = args.difficulty
    code_dir = args.code_dir
    gap_th = args.gap_th
    code_str = args.code_str

    if args.milp_type == 'code':
        if difficulty == 'all':
            milp_code_list = [os.path.join(code_dir, f'milp_{idx}-{diff}.py') 
                            for idx in range(args.code_start_idx, args.code_end_idx) 
                            for diff in ['easy', 'medium', 'hard']]
        else:
            milp_code_list = [os.path.join(code_dir, f'milp_{idx}-{difficulty}.py') for idx in range(args.code_start_idx, args.code_end_idx)]

        milp_code_list = [milp_filename for milp_filename in milp_code_list if os.path.exists(milp_filename)]
    else:
        milp_code_list = sorted(glob.glob(os.path.join(args.code_dir, '*.py')))
        if args.code_start_idx >= len(milp_code_list):
            print(f'code_start_idx {args.code_start_idx} is greater than the number of code files {len(milp_code_list)}.')
            sys.exit(0)

        code_start_idx, code_end_idx = max(0, args.code_start_idx), min(len(milp_code_list), args.code_end_idx)
        milp_code_list = milp_code_list[code_start_idx:code_end_idx]
    
    print(f'milp_code_list length: {len(milp_code_list)}')

    for idx, milp_filename in enumerate(milp_code_list):
        if not os.path.exists(milp_filename):
            print(f'MILP code file {milp_filename} does not exist. Skip.')
            continue
        print(f'---------------------------------- [{idx+1}/{len(milp_code_list)}] Generate instances for milp code::: {milp_filename} ----------------------------------')

        start_time = time.time()
        generate_instances(milp_filename, instances_save_dir, n_instances=n_instances, optimize=optimize, 
                            gap_th=gap_th, valid_timeout_duration=valid_timeout_duration, 
                            timeout_duration=timeout_duration, subprocess_timeout_duration=subprocess_timeout_duration,
                            remove_tmp=remove_tmp, n_cpus=n_cpus, code_str=code_str)
        print(f'====================================== [{idx+1}/{len(milp_code_list)}] Time taken to generate instances for {milp_filename}: {time.time() - start_time} seconds. ==================================')
    