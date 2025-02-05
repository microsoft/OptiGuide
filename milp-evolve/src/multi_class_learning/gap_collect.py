import os
import glob
import random
import argparse
import numpy as np
import pyscipopt as pyopt
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
from multiprocessing import cpu_count as _cpu_count
from gap_context import getContext
from utils import parallel_fn, save_json, load_json, save_gzip, get_train_val_test_splits, get_collect_paths
from utils import replace_extension, set_seed, get_instances_train_val_test_split, get_files_by_extension
import multiprocessing as mp

path_to_randomness_control_set = 'SCIP_settings/randomness_control.set'


class MyEventHandler(Eventhdlr):
    def __init__(self):
        super(MyEventHandler, self).__init__()
        self.first_lp_solved = False
        self.LP_val = None
        self.context = None

    def eventinit(self):
        # Catch LP events
        self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexit(self):
        # Drop LP events
        self.model.dropEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.FIRSTLPSOLVED and not self.first_lp_solved:
            self.first_lp_solved = True
            # You can add your custom logic here
            print("First LP solve event caught.")
            # lp_solution = self.model.getBestSol()
            self.LP_val = self.model.getLPObjVal()
            self.context = getContext(self.model)


def get_gap_data(model, event_handler):
    context = event_handler.context
    lp_value = event_handler.LP_val


    try:
        dual_value = model.getDualbound()
        obj_value = model.getObjVal() 
        lp_ip_gap = abs((abs(obj_value) - abs(lp_value)) / (abs(obj_value)+1e-5))
        lp_ip_gap_v2 = max(abs(obj_value), abs(lp_value)) / (1e-6 + min(abs(obj_value), abs(lp_value)))
    except:
        return False, None, {}

    n_nodes = model.getNNodes()
    solve_time = model.getSolvingTime()

    # number of variables: continuous, integer, binary, all, and the number of constraints
    n_vars = model.getNVars()
    n_bin_vars = model.getNBinVars()
    n_int_vars = model.getNIntVars()
    n_cont_vars = n_vars - n_bin_vars - n_int_vars
    n_cons = model.getNConss()

    # additional statistics
    presolve_time = model.getPresolvingTime()
    n_cuts = model.getNCutsFound()
    n_applied_cuts = model.getNCutsApplied()
    frac_applied_cuts = n_applied_cuts / n_cuts if n_cuts > 0 else 0
    final_gap = model.getGap()

    label = {'lp_ip_gap': lp_ip_gap, 'lp_ip_gap_v2': lp_ip_gap_v2, 
             'solve_time': solve_time, 'n_nodes': n_nodes, 
             'frac_applied_cuts': frac_applied_cuts, 'presolve_time': presolve_time, 
             'lp_value': lp_value, 'dual_value': dual_value, 'obj_value': obj_value, 
             'n_cuts': n_cuts, 'n_applied_cuts': n_applied_cuts, 'final_gap': final_gap}

    print(f'[final gap = {final_gap*100:.2f}%]', 'N vars::', n_vars, 'N bin vars::', n_bin_vars, 
          'N int vars::', n_int_vars, 'N cont vars::', n_cont_vars, 'N cons::', n_cons)
    print('Labels::', f'lp_ip_gap {100*lp_ip_gap:.2f}%', f'lp_ip_gap_v2 {100*lp_ip_gap_v2:.2f}%',
          f'solve_time {solve_time:.2f}', 'n_nodes', n_nodes, 
          f'frac_applied_cuts {frac_applied_cuts:.2f}', f'presolve_time {presolve_time:.2f}', 
          f'lp_value {lp_value:.2f}', f'dual_value {dual_value:.2f}', f'objective_value {obj_value:.2f}',
          f'n_cuts {n_cuts}', f'n_applied_cuts {n_applied_cuts}')
    return True, context, label


def solve(path_to_problem, time_limit=600, verbose=False, redirectOutput=False):
    model = pyopt.Model()
    event_handler = MyEventHandler()
    model.includeEventhdlr(event_handler, "MyEventHandler", "Catches the state after the first LP solve")
    model.hideOutput(not verbose)
    try:
        model.readProblem(path_to_problem)
    except:
        return False, None, {}
    model.readParams(path_to_randomness_control_set)
    if redirectOutput:
        model.redirectOutput()
    model.setParam('limits/time', time_limit)
    model.optimize()
    success, context, label = get_gap_data(model, event_handler=event_handler)
    return success, context, label


def multiprocess_solve_helper(helper_args):
    process_num, path_to_problem, time_limit, data_file, label_file = helper_args
    print(f"process {process_num}: solving problem {path_to_problem}...")
    
    try:
        success, context, label = solve(path_to_problem, time_limit=time_limit)

        if success:
            save_gzip(data_file, (context, label))
            save_json(label_file, label)
            print(f'Success! Saved {path_to_problem} ...')
        else:
            print(f'Failed to solve {data_file} ...')

        label_str = '{' + ', '.join([f'{k}: {v:.2f}' for k, v in label.items()]) + '}'
        print(f"process {process_num} - problem {path_to_problem}: success {success}" + (f" label {label_str}" if success else ""))
    except Exception as e:
        print(f"Process {process_num} encountered an unexpected error: {e}")
        return 

def worker(task_queue):
    while True:
        helper_args = task_queue.get()
        if helper_args is None:
            break
        process_num = helper_args[0]

        try:
            multiprocess_solve_helper(helper_args)
        except Exception as e:
            print(f"Process {process_num} encountered an unexpected error: {e}")


def run_multiprocess(helper_args_list, num_processes):
    with mp.Manager() as manager:
        task_queue = manager.Queue()

        # Populate the task queue
        for args in helper_args_list:
            task_queue.put(args)  # last two are label saves

        # Signal end of tasks
        for _ in range(num_processes):
            task_queue.put(None)

        processes = [mp.Process(target=worker, args=(task_queue, )) for _ in range(num_processes)]
        
        for p in processes:
            p.start()

        for p in processes:
            p.join()


if __name__ == '__main__':
    ## input instance and seed control file
    parser = argparse.ArgumentParser(description="Process configuration settings for data handling and processing.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n_cpus", type=int, default=24, help="Number of CPUs to use.")
    parser.add_argument("--parent_data_dir", type=str, default='save_dir/gap_data', help="Directory for parent data.")
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps', help="Directory for instance data.")
    parser.add_argument("--parent_data_metadata_dir", type=str, default='save_dir/gap_data/metadata', help="Directory for data metadata.")
    parser.add_argument("--parent_instances_metadata_dir", type=str, default='save_dir/instances/metadata', help="Directory for instance metadata.")

    parser.add_argument("--ntrain_instances", type=int, default=1000, help="Number of training instances.")
    parser.add_argument("--nval_instances", type=int, default=100, help="Number of validation instances.")
    parser.add_argument("--ntest_instances", type=int, default=100, help="Number of test instances.")
    parser.add_argument("--select_option", type=str, default='first', choices=['first', 'random'], help="Option to select data samples from instances.")

    parser.add_argument("--difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")
  
    parser.add_argument("--time_limit", type=int, default=300, help="Time limit for SCIP solver.")

    args = parser.parse_args()
    parent_data_dir = args.parent_data_dir
    parent_instances_dir = args.parent_instances_dir
    n_cpus = min(args.n_cpus, _cpu_count()) if args.n_cpus else _cpu_count()
    Ntrain_instances = args.ntrain_instances
    Nval_instances = args.nval_instances
    Ntest_instances = args.ntest_instances
    select_option = args.select_option
    set_seed(args.seed)

  
    instances_dir_list, generated_json_paths, instances_split_files = get_collect_paths(args)
    paths_to_problems = []
    for i_instances, (instances_dir, generated_json_path, instances_split_file) in enumerate(zip(instances_dir_list, generated_json_paths, instances_split_files)):
        print(f'[{i_instances+1} / {len(instances_dir_list)}] Collect data for {instances_dir} ...')
        # randomly sample data_max_samples for train and val
        train_split, val_split, test_split = get_instances_train_val_test_split(instances_split_file, parent_instances_dir, instances_dir, select_option=select_option,
                                                                                Ntrain_instances=Ntrain_instances, Nval_instances=Nval_instances, 
                                                                                Ntest_instances=Ntest_instances)
        if len(train_split) == 0 and len(val_split) == 0 and len(test_split) == 0:
            print(f'No instances found for {instances_dir}. Skip ...')
            continue

        print(f'# of selected train instances:: {len(train_split)} # val instances:: {len(val_split)} # test instances:: {len(test_split)}')

        # parallel data collection parameter arguments
        paths_to_problems.extend(train_split + val_split + test_split)

    # collect data
    sel_paths_to_problems = [] 
    sel_save_data_files = []
    sel_save_label_files = []
    for path in paths_to_problems:
        basename = os.path.basename(path)
        instance_dir = os.path.dirname(path).replace(parent_instances_dir, '').replace(basename, '').lstrip('/') 
        instance_idx = replace_extension(basename)
        # data_file = os.path.join(parent_data_dir, instance_dir, f'data_{instance_idx}.pkl')
        data_file = os.path.join(parent_data_dir, instance_dir, f'data_{instance_idx}.pkl.gz')
        label_file = os.path.join(parent_data_dir, instance_dir, f'label_{instance_idx}.json')
        if not os.path.exists(data_file):
            sel_paths_to_problems.append(path)
            sel_save_data_files.append(data_file)
            sel_save_label_files.append(label_file)
                

    tasks = list(zip(range(len(sel_paths_to_problems)), 
                            sel_paths_to_problems, 
                            [args.time_limit]*len(sel_paths_to_problems),
                            sel_save_data_files,
                            sel_save_label_files))
    
    print(f'# tasks to solve {len(tasks)}')
    if len(tasks) > 0:
        results = run_multiprocess(tasks, n_cpus)