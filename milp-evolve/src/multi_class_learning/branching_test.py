import os
import argparse
import ecole
import numpy as np
import torch
import torch.nn.functional as F
from branching_model import GNNPolicy
from utils import gmean, load_json, save_json, set_seed, load_checkpoint, print_dash_str, get_rollout_paths_by_dir
from utils import parallel_fn, replace_extension, get_instances_dir_train_val_test_split
from utils import get_files_by_extension, instance_numerical_sort_key
import torch.multiprocessing as mp
import time


TEST_NB_NODE_TH = 1  # skip instance where default has less than 1 nodes

def evaluate_i(args):
    instance_idx, instance_path, policy, scip_parameters, stats_filename, edge_nfeats, device, seed = args
    
    gnn_env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
            "pd": ecole.reward.PrimalDualIntegral().cumsum(),
        },
        scip_params=scip_parameters,
    )
    default_env = ecole.environment.Configuring(
        observation_function=None,
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
            "pd": ecole.reward.PrimalDualIntegral().cumsum(),
        },
        scip_params=scip_parameters,
    )

    # set seed
    gnn_env.seed(seed)
    default_env.seed(seed)

    # Accumulators for statistics
    print(f'Run SCIPs default brancher for {instance_path}...')
    try:
        default_env.reset(instance_path)
    except:
        return 
    # default_env.as_pyscipopt().hideOutput(False)
    _, _, _, _, default_info = default_env.step({})

    pyscipopt_model = default_env.model.as_pyscipopt()
    # stats
    nb_nodes_default = default_info["nb_nodes"]

    if int(nb_nodes_default) <= TEST_NB_NODE_TH:  
        print(f'Instance {instance_idx: >3} | SCIP default only have {int(nb_nodes_default)} nodes, skipping instance ...')
        return
    
    time_default = default_info["time"]
    pd_default = default_info["pd"]
    gap_default = pyscipopt_model.getGap()

    print(f'Run the GNN Brancher for {instance_path}...')
    observation, action_set, _, done, info = gnn_env.reset(instance_path)
    model_time = 0
    while not done:
        with torch.no_grad():
            constraint_features = observation.row_features
            edge_indices = observation.edge_features.indices
            edge_features = np.expand_dims(observation.edge_features.values, axis=-1)
            if edge_nfeats == 2:
                edge_features_norm = edge_features / np.linalg.norm(edge_features) 
                edge_features = np.concatenate((edge_features, edge_features_norm), axis=-1)
            variable_features = observation.variable_features

            observation = (
                torch.from_numpy(constraint_features.astype(np.float32)).to(device),
                torch.from_numpy(edge_indices.astype(np.int64)).to(device),
                torch.from_numpy(edge_features.astype(np.float32)).view(-1, edge_nfeats).to(device),
                torch.from_numpy(variable_features.astype(np.float32)).to(device),
            )
            start_time = time.time()
            logits = policy(*observation)
            model_time += time.time() - start_time
            action = action_set[logits[action_set.astype(np.int64)].argmax()]
            observation, action_set, _, done, info = gnn_env.step(action)
    
    pyscipopt_model = gnn_env.model.as_pyscipopt()
    # stats
    nb_nodes_gnn = info["nb_nodes"]
    time_gnn = info["time"]
    pd_gnn = info["pd"]
    gap_gnn = pyscipopt_model.getGap()

    # save stats to json
    instance_results = {
        'Nodes_GNN': nb_nodes_gnn,
        'Time_GNN': time_gnn, 
        'Time_GNN_Exclude_Model': max(time_gnn - model_time, 1e-5), 
        'Model_Time': model_time,
        'PD_GNN': pd_gnn,
        'Gap_GNN': gap_gnn,
        'Nodes_Default': nb_nodes_default,
        'Time_Default': time_default,
        'PD_Default': pd_default,
        'Gap_Default': gap_default
    }
    save_json(stats_filename, instance_results)
    print(f'Instance {instance_idx} [Model time {model_time:.2f}]: {instance_path}')


    nb_nodes_gain = 100 * (1 - nb_nodes_gnn / nb_nodes_default)
    time_gain = 100 * (1 - time_gnn / time_default)
    time_gain_exclude_model = 100 * (1 - (time_gnn - model_time) / time_default)
    pd_gain = 100 * (1 - pd_gnn / pd_default)
    gap_gain = 0 if gap_default < 1e-5 and gap_gnn < 1e-5 else 100 * (1 - gap_gnn / gap_default) if gap_default > 1e-5 else -100
    print(f"Instance {instance_idx: >3} | SCIP nb nodes {int(nb_nodes_default): >4d}  | SCIP time {time_default: >12.2e} | SCIP PD {pd_default: >12.2e} | SCIP Gap {gap_default: >12.2e} | SCIP time {time_default: >12.2e}")
    print(f"             | GNN  nb nodes {int(nb_nodes_gnn): >4d}  | GNN  time {time_gnn: >12.2e} | GNN PD {pd_gnn: >12.2e}  | GNN Gap {gap_gnn: >12.2e} | Model Time {model_time: >12.2e}")
    print(f"             | Gain     {nb_nodes_gain: >8.2f}%  | Gain         {time_gain: >8.2f}% |"
          f" Gain       {pd_gain: >8.2f}% | Gain       {gap_gain: >8.2f}% | Gain (Excl. Model Time) {time_gain_exclude_model: >8.2f}%")
    

def worker(task_queue):
    while True:
        helper_args = task_queue.get()
        if helper_args is None:
            break
        try:
            evaluate_i(helper_args)
        except Exception as e:
            print(f"Encountered an unexpected error: {e}")


def run_multiprocess(helper_args_list, num_processes):
    task_queue = mp.Queue()

    # Populate the task queue
    for args in helper_args_list:
        task_queue.put(args)

    # Signal end of tasks
    for _ in range(num_processes):
        task_queue.put(None)

    processes = [mp.Process(target=worker, args=(task_queue,)) for _ in range(num_processes)]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def evaluate(policy, instance_paths, parent_instances_dir, scip_parameters, parent_test_stats_dir, device, 
             do_parallel=False, n_cpus=24, reevaluate=False, edge_nfeats=1, seed=1):
    tasks = []        
    for i, instance_path in enumerate(instance_paths):
        if not os.path.exists(instance_path):
            print(f"Instance {instance_path} does not exist.")
            continue
        basename = os.path.basename(instance_path)   
        instance_dir = os.path.dirname(instance_path).replace(parent_instances_dir, '').replace(basename, '').lstrip('/') 
        instance_idx = replace_extension(basename) 

        stats_filename = os.path.join(parent_test_stats_dir, instance_dir, f'stats_{instance_idx}.json')

        if reevaluate or not os.path.exists(stats_filename):
            tasks.append((i, instance_path, policy, scip_parameters, stats_filename, edge_nfeats, device, seed))

    if do_parallel and n_cpus and n_cpus > 1:
        # policy.share_memory()
        mp.set_start_method('spawn', force=True)
        run_multiprocess(tasks, n_cpus)
        # parallel_fn(evaluate_i, tasks, n_cpus=n_cpus)
    else:
        for args in tasks:
            evaluate_i(args)


def load_and_summarize_performance(instance_paths, parent_instances_dir, parent_test_stats_dir, do_gmean=True):
    

    def compute_averages_and_gains(results, method_name):
        avg_nb_nodes = gmean(results[f'Nodes_{method_name}'], 0) if do_gmean else np.median(results[f'Nodes_{method_name}'])
        avg_time = gmean(results[f'Time_{method_name}']) if do_gmean else np.median(results[f'Time_{method_name}'])
        if method_name == 'GNN':
            avg_time_exclude_model = gmean(results[f'Time_GNN_Exclude_Model']) if do_gmean else np.median(results[f'Time_GNN_Exclude_Model'])
        else:
            avg_time_exclude_model =  avg_time
        avg_pd = gmean(results[f'PD_{method_name}']) if do_gmean else np.median(results[f'PD_{method_name}'])
        avg_gap = gmean(results[f'Gap_{method_name}']) if do_gmean else np.median(results[f'Gap_{method_name}'])
        num_instances = len(results[f'Nodes_{method_name}'])
        return avg_nb_nodes, avg_time, avg_time_exclude_model, avg_pd, avg_gap, num_instances
    

    instance_results = {
        'Nodes_GNN': [],
        'Time_GNN': [],
        'PD_GNN': [],
        'Gap_GNN': [],
        'Time_GNN_Exclude_Model': [],
        'Nodes_Default': [],
        'Time_Default': [],
        'PD_Default': [],
        'Gap_Default': [],
    }

    for instance_path in instance_paths:
        basename = os.path.basename(instance_path)   
        instance_dir = os.path.dirname(instance_path).replace(parent_instances_dir, '').replace(basename, '').lstrip('/') 
        instance_idx = replace_extension(basename)
        filename = os.path.join(parent_test_stats_dir, instance_dir, f'stats_{instance_idx}.json')
        if os.path.exists(filename):
            result = load_json(filename)
            for key in instance_results.keys():
                if key == 'Time_GNN_Exclude_Model':
                    result[key] = max(result['Time_GNN'] - result['Model_Time'], 1e-5)
                instance_results[key].append(result[key])

    methods = ['Default', 'GNN']  # 'Pseudocost', 
    averages = {}

    for method in methods:
        if instance_results[f'Nodes_{method}']:
            averages[method] = compute_averages_and_gains(instance_results, method)
    
    # Printing header
    print(f"{'Method':>15}{'Average Nodes':>20}{'Average Time':>20}{'Avg. Time (Exc. Model)':>20}{'Average PD':>20}{'Average Gap':>20}{'Num Instances':>20}")
    print('-------------------------------------------------------------------------------------------------------------------------------------------')
    for method in methods:
        if method in averages:
            avg_nb_nodes, avg_time, avg_time_exclude_model, avg_pd, avg_gap, num_instances = averages[method]
            print(f"{method:>15}{avg_nb_nodes:>20.2f}{avg_time:>20.2f}{avg_time_exclude_model:>20.2f}{avg_pd:>20.2e}{avg_gap:>20.02f}{num_instances:>15}")
    print('===========================================================================================================================================')
    
    # Gain calculations
    if 'Default' in averages and 'GNN' in averages:
        avg_nb_nodes_default, avg_time_default, _, avg_pd_default, avg_gap_default, _ = averages['Default']
        avg_nb_nodes_gnn, avg_time_gnn, avg_time_gnn_exclude_model, avg_pd_gnn, avg_gap_gnn, _ = averages['GNN']
        nb_nodes_gain = 100 * (1 - avg_nb_nodes_gnn/avg_nb_nodes_default)
        time_gain = 100 * (1 - avg_time_gnn/avg_time_default)
        time_gain_exclude_models = 100 * (1 - avg_time_gnn_exclude_model/avg_time_default)
        pd_gain = 100 * (1 - avg_pd_gnn/avg_pd_default)
        gap_gain = 0 if avg_gap_default < 1e-5 and avg_gap_gnn < 1e-5 else 100 * (1 - avg_gap_gnn/avg_gap_default) if avg_gap_default > 1e-5 else -100
        print(f"{'Gain (GNN)':>15}{nb_nodes_gain:>19.2f}%{time_gain:>19.2f}%{time_gain_exclude_models:>19.2f}{pd_gain:>19.2f}%{gap_gain:>19.2f}%")
    print('===========================================================================================================================================')
    

def load_and_summarize_performance_separate(instance_paths, parent_instances_dir, parent_test_stats_dir, do_gmean_zero_gap=True, do_gmean_nonzero_gap=False):
    
    def compute_averages_and_gains(results, method_name, do_gmean=True):
        avg_nb_nodes = gmean(results[f'Nodes_{method_name}'], 0) if do_gmean else np.median(results[f'Nodes_{method_name}'])
        avg_time = gmean(results[f'Time_{method_name}']) if do_gmean else np.median(results[f'Time_{method_name}'])
        if method_name == 'GNN':
            avg_time_exclude_model = gmean(results[f'Time_GNN_Exclude_Model']) if do_gmean else np.median(results[f'Time_GNN_Exclude_Model'])
        else:
            avg_time_exclude_model = avg_time
        avg_pd = gmean(results[f'PD_{method_name}']) if do_gmean else np.median(results[f'PD_{method_name}'])
        avg_gap = gmean(results[f'Gap_{method_name}']) if do_gmean else np.median(results[f'Gap_{method_name}'])
        num_instances = len(results[f'Nodes_{method_name}'])
        return avg_nb_nodes, avg_time, avg_time_exclude_model, avg_pd, avg_gap, num_instances

    def compute_gain(default_avg, gnn_avg):
        return 100 * (1 - gnn_avg / default_avg) if default_avg > 1e-5 else 0

    instance_results = {
        'Nodes_GNN': [],
        'Time_GNN': [],
        'PD_GNN': [],
        'Gap_GNN': [],
        'Time_GNN_Exclude_Model': [],
        'Nodes_Default': [],
        'Time_Default': [],
        'PD_Default': [],
        'Gap_Default': [],
    }

    # Separate instance results for non-zero gap and zero gap instances
    instance_results_nonzero_gap = {key: [] for key in instance_results.keys()}
    instance_results_zero_gap = {key: [] for key in instance_results.keys()}

    for instance_path in instance_paths:
        basename = os.path.basename(instance_path)   
        instance_dir = os.path.dirname(instance_path).replace(parent_instances_dir, '').replace(basename, '').lstrip('/') 
        instance_idx = replace_extension(basename)
        filename = os.path.join(parent_test_stats_dir, instance_dir, f'stats_{instance_idx}.json')
        if os.path.exists(filename):
            result = load_json(filename)
            for key in instance_results.keys():
                if key == 'Time_GNN_Exclude_Model':
                    result[key] = max(result['Time_GNN'] - result['Model_Time'], 1e-5)
                instance_results[key].append(result[key])
                if result['Gap_Default'] > 1e-5 or result['Gap_GNN'] > 1e-5:
                    instance_results_nonzero_gap[key].append(result[key])
                else:
                    instance_results_zero_gap[key].append(result[key])

    methods = ['Default', 'GNN']
    averages = {}
    averages_nonzero_gap = {}
    averages_zero_gap = {}

    # Compute averages and gains for all instances
    for method in methods:
        if instance_results[f'Nodes_{method}']:
            averages[method] = compute_averages_and_gains(instance_results, method, do_gmean=do_gmean_nonzero_gap)
        if instance_results_nonzero_gap[f'Nodes_{method}']:
            averages_nonzero_gap[method] = compute_averages_and_gains(instance_results_nonzero_gap, method, do_gmean=do_gmean_nonzero_gap)
        if instance_results_zero_gap[f'Nodes_{method}']:
            averages_zero_gap[method] = compute_averages_and_gains(instance_results_zero_gap, method, do_gmean=do_gmean_zero_gap)
    
    def print_summary(averages, label):
        print(f"Summary for {label} Instances")
        print(f"{'Method':>15}{'Average Nodes':>20}{'Average Time':>20}{'Avg. Time (Exc. Model)':>20}{'Average PD':>20}{'Average Gap':>20}{'Num Instances':>20}")
        print('-------------------------------------------------------------------------------------------------------------------------------------------')
        for method in methods:
            if method in averages:
                avg_nb_nodes, avg_time, avg_time_exclude_model, avg_pd, avg_gap, num_instances = averages[method]
                print(f"{method:>15}{avg_nb_nodes:>20.2f}{avg_time:>20.2f}{avg_time_exclude_model:>20.2f}{avg_pd:>20.2e}{avg_gap:>20.03f}{num_instances:>15}")
        print('===========================================================================================================================================')
    
    def print_gains(averages):
        if 'Default' in averages and 'GNN' in averages:
            avg_nb_nodes_default, avg_time_default, _, avg_pd_default, avg_gap_default, _ = averages['Default']
            avg_nb_nodes_gnn, avg_time_gnn, avg_time_gnn_exclude_model, avg_pd_gnn, avg_gap_gnn, _ = averages['GNN']
            nb_nodes_gain = compute_gain(avg_nb_nodes_default, avg_nb_nodes_gnn)
            time_gain = compute_gain(avg_time_default, avg_time_gnn)
            time_gain_exclude_models = compute_gain(avg_time_default, avg_time_gnn_exclude_model)
            pd_gain = compute_gain(avg_pd_default, avg_pd_gnn)
            gap_gain = 0 if avg_gap_default < 1e-5 and avg_gap_gnn < 1e-5 else compute_gain(avg_gap_default, avg_gap_gnn)
            print(f"{'Gain (GNN)':>15}{nb_nodes_gain:>19.2f}%{time_gain:>19.2f}%{time_gain_exclude_models:>19.2f}{pd_gain:>19.3f}%{gap_gain:>19.2f}%")
        print('===========================================================================================================================================')

    # Print summaries and gains for all instances, non-zero gap instances, and zero gap instances
    # print_summary(averages, "All")
    # print_gains(averages)
    
    print_summary(averages_nonzero_gap, "Non-Zero Gap")
    print_gains(averages_nonzero_gap)
    
    print_summary(averages_zero_gap, "Zero Gap")
    print_gains(averages_zero_gap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for PyTorch model training.')

    # General settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument("--n_cpus", type=int, default=50, help="Number of CPUs to use.")
    parser.add_argument('--parent_model_dir', type=str, default='save_dir/branching_checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--parent_test_stats_dir', type=str, default='save_dir/branching_test_stats', help='Directory to save test stats.')
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps', help="Directory for instance data.")
    parser.add_argument("--parent_instances_metadata_dir", type=str, default='save_dir/instances/metadata', help="Directory for instance metadata.")

    parser.add_argument("--max_data_per_instance", type=int, default=50, help="Maximum number of data samples per instance.")
    parser.add_argument("--ntrain_instances", type=int, default=0, help="Number of training instances to collect data.")
    parser.add_argument("--nval_instances", type=int, default=0, help="Number of validation instances to collect data.")
    parser.add_argument("--ntest_instances", type=int, default=10, help="Number of test instances to evaluate on.")
    parser.add_argument("--select_option", type=str, default='first', choices=['first', 'random'], help="Option to select data samples from instances.")

    # model parameters for getting the correct model name
    parser.add_argument("--model_difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--model_code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--model_code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--model_ntrain_instances", type=int, default=35, help="Number of training instances to collect data.")
    parser.add_argument("--model_max_data_per_instance", type=int, default=50, help="Maximum number of data samples per instance.")
    parser.add_argument("--model_code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--model_code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--model_ntrain_data", type=int, default=float('inf'), help="Number of training instances to collect data.")
    parser.add_argument("--model_code_str", type=str, default='code', help="String to identify the code instances.")
    # GNN arguments
    parser.add_argument("--emb_size", type=int, default=64, help="Embedding size of the GNN.")
    parser.add_argument("--edge_nfeats", type=int, default=1, help="Number of edge features.", choices=[1, 2])
    parser.add_argument("--n_layers", type=int, default=1, help="Number of GNN layers.")

    parser.add_argument("--test_instances_dir_split_file", type=str, default='', help="File containing the instance directories to split.")
    parser.add_argument("--reevaluate", action="store_true", help="Flag to reevaluate instances.")
    # test parameters
    parser.add_argument("--difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")

    parser.add_argument("--time_limit", type=int, default=360, help="Time limit for SCIP solver.")
    parser.add_argument("--eval_split", type=str, default='test', choices=['train', 'test', 'val', 'all'], help="Split to evaluate.")
    parser.add_argument("--not_test_by_dir", action="store_true", help="Flag to determine whether to test by instance directories.")
    parser.add_argument("--load_model_dir", type=str, default='', help="Model directory to load.")
    parser.add_argument("--model_suffix", type=str, default='', help="Suffix to add to the model directory.")
    
    args = parser.parse_args()

    do_parallel = True
    seed = args.seed
    N_CPUS = args.n_cpus
    Ntrain_instances = args.ntrain_instances
    Nval_instances = args.nval_instances
    Ntest_instances = args.ntest_instances
    select_option = args.select_option

    parent_model_dir = args.parent_model_dir
    parent_instances_dir = args.parent_instances_dir
    parent_instances_metadata_dir = args.parent_instances_metadata_dir
    parent_test_stats_dir = args.parent_test_stats_dir
    # GNN parameters
    emb_size = args.emb_size
    edge_nfeats = args.edge_nfeats
    n_layers = args.n_layers
    test_by_dir = not args.not_test_by_dir


    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(parent_test_stats_dir, exist_ok=True)

    #### Load Train, Val, Test Data split
    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": args.time_limit,
    }

    policy = GNNPolicy(emb_size=emb_size, edge_nfeats=edge_nfeats, n_layers=n_layers).to(device)
    

    # print the number of parametesr that the model has
    print(f"Number of parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # follow the train set split of the instances to load the test set
    instances_dir_list, instances_dir_split_file, model_dir, parent_test_stats_dir = get_rollout_paths_by_dir(args, parser)
    if args.test_instances_dir_split_file:
        assert os.path.exists(args.test_instances_dir_split_file), f"Test instances dir split file {args.test_instances_dir_split_file} does not exist."
        print(f'Loading test_instances_dir_split_file from {args.test_instances_dir_split_file}...')
        instances_dir_split_file = args.test_instances_dir_split_file
        dir_file = load_json(instances_dir_split_file)
        instances_dir_list = dir_file['train'] + dir_file['val'] + dir_file['test']

    if args.load_model_dir != '':
        model_dir = args.load_model_dir 
        modelname = os.path.basename(model_dir)
    else:
        modelname = os.path.basename(model_dir)

        if args.model_suffix:
            model_dir = model_dir + f'_{args.model_suffix}'
            modelname = modelname + f'_{args.model_suffix}'
    
    parent_test_stats_dir = os.path.join(args.parent_test_stats_dir, modelname, 'by_dir')

    assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"

    load_checkpoint(policy, None, step='max', save_dir=model_dir, device=device)

    policy.eval()
    policy.to(device)
    
    parent_test_stats_dir = os.path.join(parent_test_stats_dir, args.eval_split)
    os.makedirs(parent_test_stats_dir, exist_ok=True)
    
    train_instances_dir_split, val_instances_dir_split, test_instances_dir_split = get_instances_dir_train_val_test_split(instances_dir_split_file, instances_dir_list)

    if args.eval_split == 'all':
        instances_dir_split = train_instances_dir_split + val_instances_dir_split + test_instances_dir_split
    else:
        instances_dir_split = test_instances_dir_split if args.eval_split == 'test' else val_instances_dir_split if args.eval_split == 'val' else train_instances_dir_split

    instances_dir_split = [instance_dir for instance_dir in instances_dir_split 
                            if args.code_start_idx <= int(os.path.basename(instance_dir).split('_')[1].split('-')[0]) <= args.code_end_idx]
    
    if len(instances_dir_split) == 0:
        print(f'No {args.eval_split} instance directories found in {instances_dir_split_file}. Please check the instances_dir_split_file.')
        exit()

    print(f'{args.eval_split}: {len(instances_dir_split)} to check::', instances_dir_split, '...')

    test_files_list = []

    for i_dir, instances_dir in enumerate(instances_dir_split):  #  tqdm.tqdm(instances_dir_split):            
        instances_split_file = os.path.join(parent_instances_metadata_dir, instances_dir, f'instances_split.json')
        if not os.path.exists(instances_split_file):
            test_files = get_files_by_extension(os.path.join(parent_instances_dir, instances_dir))                   
        else:
            test_files_all = load_json(instances_split_file)
            test_files = test_files_all['train'] + test_files_all['val'] + test_files_all['test']
            test_files = sorted(test_files, key=instance_numerical_sort_key)[:Ntest_instances]
        test_files_list.append(test_files)

        print(f'[{i_dir+1}/{len(instances_dir_split)}] {instances_dir}:: Number of test instances: {len(test_files)}, evaluate the first {Ntest_instances} instances.')
        if test_by_dir:
            print_dash_str(f'Evaluating {len(test_files)} test files for {instances_dir}. Save stats to {parent_test_stats_dir} ...')
            evaluate(policy, test_files, parent_instances_dir, scip_parameters, parent_test_stats_dir, device, 
                    do_parallel=do_parallel, n_cpus=N_CPUS, reevaluate=args.reevaluate, edge_nfeats=edge_nfeats,
                    seed=seed)
            load_and_summarize_performance(test_files, parent_instances_dir, parent_test_stats_dir)

    test_files_all = [x for test_files in test_files_list for x in test_files]
    if test_by_dir:
        print_dash_str(f'Overall performance for {len(test_files_all)} test files from all instances directories...')
    else:
        print_dash_str(f'Evaluating {len(test_files_all)} test files from all instances directories...')
        evaluate(policy, test_files_all, parent_instances_dir, scip_parameters, parent_test_stats_dir, device, 
                    do_parallel=do_parallel, n_cpus=N_CPUS, reevaluate=args.reevaluate, edge_nfeats=edge_nfeats,
                    seed=seed)
        
        
    print_dash_str(f'Individual class performances')
    for instances_dir, test_files in zip(instances_dir_split, test_files_list):
        print_dash_str(f'Performance for {instances_dir}')
        load_and_summarize_performance(test_files, parent_instances_dir, parent_test_stats_dir)

    print_dash_str(f'Overall performance for all {len(instances_dir_split)} instances directories')
    # load_and_summarize_performance(test_files_all, parent_instances_dir, parent_test_stats_dir)
    load_and_summarize_performance_separate(test_files_all, parent_instances_dir, parent_test_stats_dir)
