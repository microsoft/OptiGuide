import os
import glob
import argparse
import pyscipopt as pyopt
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
from gap_context import getContext
import multiprocessing as mp
from utils import save_gzip, replace_extension, set_seed

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
            self.LP_val = self.model.getLPObjVal()
            self.context = getContext(self.model)

            # Interrupt the solve process
            self.model.interruptSolve()


def solve(path_to_problem, time_limit=600, verbose=False, redirectOutput=False):
    model = pyopt.Model()
    event_handler = MyEventHandler()
    model.includeEventhdlr(event_handler, "MyEventHandler", "Catches the state after the first LP solve")
    model.hideOutput(not verbose)
    try:    
        model.readProblem(path_to_problem)
    except:
        print(f'Failed to read {path_to_problem}')
        return None
    model.readParams(path_to_randomness_control_set)
    if redirectOutput:
        model.redirectOutput()
    model.setParam('limits/time', time_limit)
    model.optimize()
    
    context = event_handler.context

    return context


def multiprocess_solve_helper(helper_args):
    process_num, path_to_problem, time_limit, data_file = helper_args
    # print(f"process {process_num}: solving problem {path_to_problem}...")
    
    context = solve(path_to_problem, time_limit=time_limit)
    if context is not None:
        save_gzip(data_file, context)
        print('Success! Saved context to', data_file)
    
    # print(f"process {process_num} finished - problem {path_to_problem}")
  
    return context

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
    if num_processes == 1:
        for helper_args in helper_args_list:
            multiprocess_solve_helper(helper_args)
    else:

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
    parser.add_argument("--n_cpus", type=int, default=60, help="Number of CPUs to use.")
    parser.add_argument("--time_limit", type=int, default=150, help="Time limit for SCIP solver.")
    parser.add_argument("--parent_data_dir", type=str, default='save_dir/contrast/data', help="Parent directory to save data.")
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps/code_v1', help="Parent directory for instances")

    args = parser.parse_args()

    set_seed(args.seed)
    parent_data_dir = args.parent_data_dir
    parent_instances_dir = args.parent_instances_dir

    os.makedirs(parent_data_dir, exist_ok=True)

    n_cpus = min(args.n_cpus, mp.cpu_count()) if args.n_cpus else mp.cpu_count()
    paths_to_problems = [path for path in glob.glob(os.path.join(parent_instances_dir, '*', '*.mps.gz'))] 

    sel_paths_to_problems = [] 
    sel_save_data_files = []
    for path in paths_to_problems:
        basename = os.path.basename(path)
        instance_dir = os.path.basename(os.path.dirname(path))
        instance_name = replace_extension(os.path.basename(path)).replace('instance_', '').replace('milp_', '')
        os.makedirs(os.path.join(parent_data_dir, instance_dir), exist_ok=True)

        data_file = os.path.join(parent_data_dir, instance_dir, f'data_{instance_name}.pkl.gz')
        if not os.path.exists(data_file):
            sel_paths_to_problems.append(path)
            sel_save_data_files.append(data_file)

    tasks = list(zip(range(len(sel_paths_to_problems)), 
                            sel_paths_to_problems, 
                            [args.time_limit]*len(sel_paths_to_problems),
                            sel_save_data_files))
    
    print(f'--------------  # tasks to solve {len(tasks)} out of {len(paths_to_problems)}  -----------------')

    if len(tasks) > 0:
        run_multiprocess(tasks, n_cpus)

    num_gen = len([data_file for data_file in sel_save_data_files if os.path.exists(data_file)])
    print(f'--------------  Generated {num_gen} contexts out of {len(sel_save_data_files)} instances.  ---------------')
