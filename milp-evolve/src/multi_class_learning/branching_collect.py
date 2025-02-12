import os
import glob
import argparse
import random
import ecole
import numpy as np
from branching_model import ExploreThenStrongBranch
from ecole.instance import FileGenerator
from utils import get_collect_paths, load_json, save_json, save_gzip
from utils import parallel_fn, replace_extension, set_seed, get_instances_train_val_test_split, branching_data_numerical_sort_key
from multiprocessing import cpu_count as _cpu_count
import multiprocessing as mp

def data_collection_i(args):
    instance_path, instance_idx, instance_dir, scip_parameters, seed, max_data_per_instance = args

    env = ecole.environment.Branching(
        observation_function=(
            ExploreThenStrongBranch(expert_probability=0.05),
            ecole.observation.NodeBipartite(),
        ),
        scip_params=scip_parameters,
    )

    # This will seed the environment for reproducibility
    env.seed(seed)
    data_sample_counter = 0
    '''Get GNN input, and also get code embedding from LLM'''
    try:
        observation, action_set, _, done, _ = env.reset(instance_path)
    except:
        print('Error in reseting the environment for instance:', instance_path)
        return 0, 0, -1

    num_steps = 0
    while not done and data_sample_counter < max_data_per_instance:
        (scores, scores_are_expert), node_observation = observation
        action = action_set[scores[action_set].argmax()]

        # Only save samples if they are coming from the expert (strong branching)
        if scores_are_expert:
            data_sample_counter += 1
            data = [node_observation, action, action_set, scores]
            filename = os.path.join(parent_data_dir, instance_dir, f'data_{instance_idx}_{data_sample_counter}.pkl')
            save_gzip(filename, data)
            # print(f"Instance {instance_idx}:: {sample_counter} samples collected so far to {os.path.join(parent_data_dir, instance_dir)} ...")

        observation, action_set, _, done, _ = env.step(action)
        num_steps += 1

    pyscipopt_model = env.model.as_pyscipopt()
    final_gap = pyscipopt_model.getGap()

    print(f"Instance {instance_idx}:: {data_sample_counter} data samples collected to {os.path.join(parent_data_dir, instance_dir)}. Final Gap {pyscipopt_model.getGap():.2f} ...")
    return data_sample_counter, num_steps, final_gap

    

def worker(task_queue, results):
    while True:
        helper_args = task_queue.get()
        if helper_args is None:
            break
        process_num = helper_args[0]

        try:
            result = data_collection_i(helper_args[1:])
        except Exception as e:
            print(f"Process {process_num} encountered an unexpected error: {e}")
            result = (0, 0, -1)

        results[process_num] = result


def run_multiprocess(helper_args_list, num_processes):
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

        return [results.get(helper_args[0], (0, 0, -1)) for helper_args in helper_args_list]
    

def data_collection(instance_paths, parent_instances_dir, scip_parameters, metadata_path, 
                    max_data_samples=10000, max_data_per_instance=200,
                    seed=0, n_cpus=None, generated_json_path=None):
    generated_instances = load_json(generated_json_path, default=[])
    metadata = load_json(metadata_path) if os.path.exists(metadata_path) else {}
    
    total_data_samples_counter = 0
    total_instances_counter = 0

    tasks = []

    process_num = 0
    for instance_path in instance_paths:
        basename = os.path.basename(instance_path)   
        instance_dir = os.path.dirname(instance_path).replace(parent_instances_dir, '').replace(basename, '').lstrip('/') 
        instance_idx = replace_extension(basename)
        if instance_path not in generated_instances:
            # instances to be processed
            tasks.append((process_num, instance_path, instance_idx, instance_dir, scip_parameters, seed, max_data_per_instance))
            process_num += 1
        else:
            # already processed
            saved_files = glob.glob(os.path.join(parent_data_dir, instance_dir, f'data_{instance_idx}_*.pkl'))
            saved_files = sorted(saved_files, key=branching_data_numerical_sort_key)[:max_data_per_instance]
            total_data_samples_counter += len(saved_files)
            total_instances_counter += 1 

    if total_data_samples_counter >= max_data_samples:
        print(f'Data collection completed:: {total_data_samples_counter} collected!')
        return

    if len(tasks) > 0:
        results = run_multiprocess(tasks, n_cpus)

        if len(results) > 0:
            data_sample_counter, num_steps, final_gap = zip(*results)

            total_data_samples_counter += sum(data_sample_counter)
            total_instances_counter += len(tasks)
            for task in tasks:
                generated_instances.append(task[0])
            
            save_json(generated_json_path, generated_instances)
            print(f'Number of instances so far:: {total_instances_counter}. Number of data samples collected so far:: {total_data_samples_counter}.')
            print(f'Data collection completed:: {total_data_samples_counter} collected, max data samples {max_data_samples}')

            for _, instance_path, instance_idx, instance_dir, _, _, _ in tasks:
                metadata[instance_path] = {'instance_idx': instance_idx, 'num_steps': num_steps, 'final_gap': final_gap}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process configuration settings for data handling and processing.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n_cpus", type=int, default=24, help="Number of CPUs to use.")
    parser.add_argument("--parent_data_dir", type=str, default='save_dir/branching_data', help="Directory for parent data.")
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps', help="Directory for instance data.")
    parser.add_argument("--parent_data_metadata_dir", type=str, default='save_dir/branching_data/metadata', help="Directory for data metadata.")
    parser.add_argument("--parent_instances_metadata_dir", type=str, default='save_dir/instances/metadata', help="Directory for instance metadata.")

    parser.add_argument("--max_data_samples", type=int, default=10000, help="Maximum number of samples to process.")
    parser.add_argument("--max_data_per_instance", type=int, default=20, help="Maximum number of data samples per instance.")
    parser.add_argument("--ntrain_instances", type=int, default=1000, help="Number of training instances.")
    parser.add_argument("--nval_instances", type=int, default=100, help="Number of validation instances.")
    parser.add_argument("--ntest_instances", type=int, default=100, help="Number of test instances.")
    parser.add_argument("--select_option", type=str, default='first', choices=['first', 'random'], help="Option to select data samples from instances.")
    
    parser.add_argument("--instances_dir_list", nargs='*', default=[], help="List of directories containing instances.")
    parser.add_argument("--by_dir_split", action="store_true", help="Flag to determine whether to split by instance directories.")
    parser.add_argument("--instances_dir_split_file", type=str, default='', help="File containing the instance directories to split.")

    parser.add_argument("--difficulty", type=str, nargs='*', default=['easy'], help="Difficulty of instances to process.")
    parser.add_argument("--code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")

    parser.add_argument("--time_limit", type=int, default=360, help="Time limit for SCIP solver.")

    args = parser.parse_args()

    max_data_samples = args.max_data_samples
    max_data_per_instance = args.max_data_per_instance
    seed = args.seed
    n_cpus = min(args.n_cpus, _cpu_count()) if args.n_cpus else _cpu_count()
    Ntrain_instances = args.ntrain_instances
    Nval_instances = args.nval_instances
    Ntest_instances = args.ntest_instances
    select_option = args.select_option

    set_seed(seed)

    parent_data_dir = args.parent_data_dir
    parent_instances_dir = args.parent_instances_dir
    parent_instances_metadata_dir = args.parent_instances_metadata_dir
    parent_data_metadata_dir = args.parent_data_metadata_dir

    if len(args.instances_dir_list) > 0:
        instances_dir_list = args.instances_dir_list
        instances_split_files = [os.path.join(parent_instances_metadata_dir, instances_dir, f'instances_split.json') for instances_dir in instances_dir_list]
        generated_json_paths = [os.path.join(parent_data_metadata_dir, instances_dir, f'generated.json') for instances_dir in instances_dir_list]
    elif args.by_dir_split and os.path.exists(args.instances_dir_split_file):
        instances_dir_split_file = load_json(args.instances_dir_split_file)
        instances_dir_list = instances_dir_split_file['train'] + instances_dir_split_file['val'] + instances_dir_split_file['test']
        instances_split_files = [os.path.join(parent_instances_metadata_dir, instances_dir, f'instances_split.json') for instances_dir in instances_dir_list]  # data_split.json for code
        generated_json_paths = [os.path.join(parent_data_metadata_dir, instances_dir, f'generated.json') for instances_dir in instances_dir_list]
    else:
        instances_dir_list, generated_json_paths, instances_split_files = get_collect_paths(args)

    metadata_paths = [os.path.join(parent_data_metadata_dir, instances_dir, f'metadata.json') for instances_dir in instances_dir_list]
    
    for i_dir, (instances_dir, generated_json_path, metadata_path, instances_split_file) in enumerate(zip(instances_dir_list, generated_json_paths, metadata_paths, instances_split_files)):
        print(f'[{i_dir+1} / {len(instances_dir_list)}] Collect data for {instances_dir} ...')
        # for each instance split: randomly sample train, val, test splits. Collect test splits data for by_dir data split training
        train_split, val_split, test_split = get_instances_train_val_test_split(instances_split_file, parent_instances_dir, instances_dir, select_option=select_option,
                                                                                    Ntrain_instances=Ntrain_instances, Nval_instances=Nval_instances, Ntest_instances=Ntest_instances)
        if len(train_split) == 0 and len(val_split) == 0 and len(test_split) == 0:
            print(f'No instances found for {instances_dir}. Skip ...')
            continue

        print(f'[{instances_dir}] # train_split = {len(train_split)}, # val split = {len(val_split)}, # test split = {len(test_split)}')

        data_dir = os.path.join(parent_data_dir, instances_dir)
        os.makedirs(data_dir, exist_ok=True)

        instance_paths = train_split + val_split + test_split
        print(f'Collect training + val + test data for {len(instance_paths)} instances...')
     
        scip_parameters = {
            "separating/maxrounds": 0,
            "presolving/maxrestarts": 0,
            "limits/time": args.time_limit,
        }

        data_collection(instance_paths, parent_instances_dir, scip_parameters, metadata_path, 
                        max_data_samples=max_data_samples, max_data_per_instance=max_data_per_instance,
                        seed=seed, n_cpus=n_cpus, generated_json_path=generated_json_path)
