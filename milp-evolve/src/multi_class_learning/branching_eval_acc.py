import os
import glob
import pickle
import argparse
from pathlib import Path
import random
import pdb
import numpy as np
import torch
import torch.nn.functional as F
from branching_model import GNNPolicy
from utils import load_json, save_json, set_seed, load_checkpoint, print_dash_str, get_rollout_paths_by_dir
from utils import get_instances_dir_train_val_test_split
import torch.multiprocessing as mp
import time
import tqdm
from branching_dataset import GraphDataset
import torch_geometric

def evaluate(policy, data_loader, device, stats_filename):
    mean_loss = 0
    mean_acc = 0
    mean_top5_acc = 0
    mean_score_diff = 0
    mean_normalized_score_diff = 0

    policy.eval()

    n_samples_processed = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            # Index the results by the candidates, and split and pad them
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choices)
            # if isnan: pdb
            if torch.isnan(loss):
                pdb.set_trace()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()
            top5_acc = (true_scores.gather(-1, logits.topk(min(5, logits.size(-1))).indices) == true_bestscore).float().max(dim=-1).values.mean().item()

            score_diff = (true_bestscore - true_scores.gather(-1, predicted_bestindex)).abs().mean().item()
            normalized_score_diff = ((true_bestscore - true_scores.gather(-1, predicted_bestindex)) / (true_bestscore + 1e-5)).mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            mean_top5_acc += top5_acc * batch.num_graphs

            mean_score_diff += score_diff * batch.num_graphs
            mean_normalized_score_diff += normalized_score_diff * batch.num_graphs
            n_samples_processed += batch.num_graphs

            # if score_diff > 500:
            #     print(f'Debug me!!! score_diff {score_diff}.')
            #     # pdb.set_trace()

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    mean_top5_acc /= n_samples_processed
    mean_score_diff /= n_samples_processed
    mean_normalized_score_diff /= n_samples_processed


    instance_dir_results = {
        'Loss': mean_loss,
        'Accuracy': mean_acc,
        'Top5_Accuracy': mean_top5_acc,
        'Score_diff': mean_score_diff,
        'Normalized_score_diff': mean_normalized_score_diff,
        "n_samples": n_samples_processed
    }


    save_json(stats_filename, instance_dir_results)
    return mean_loss, mean_acc, mean_top5_acc, mean_score_diff, mean_normalized_score_diff


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output



def load_and_summarize_performance(data_paths, parent_data_dir, parent_test_stats_dir, do_gmean=True):
    def compute_averages(data_results):
        num_samples = sum(data_results['n_samples'])
        for key in data_results.keys():
            if key != 'n_samples':
                if len(data_results['n_samples']) > 0:
                    data_results[key] = sum([(y if (np.isnan(x) and key == 'Normalized_score_diff') else x * y) for x, y in zip(data_results[key], data_results['n_samples'])]) / num_samples
                else:
                    data_results[key] = 0
        
    data_results = {
        'Loss': [],
        'Accuracy': [],
        'Top5_Accuracy': [],
        'Score_diff': [],
        'Normalized_score_diff': [],
        'n_samples': []
    }

    unique_instance_dirs = set([os.path.dirname(data_path).replace(parent_data_dir, '').replace(os.path.basename(data_path), '').lstrip('/') for data_path in data_paths])
    for instance_dir in unique_instance_dirs:
        filename = os.path.join(parent_test_stats_dir, instance_dir, f'eval_acc.json')
        if os.path.exists(filename):
            result = load_json(filename)
            for key in data_results:
                data_results[key].append(result[key])

    compute_averages(data_results)
    num_samples = sum(data_results['n_samples'])
    
    print(f"{'Instance':>15}{'Loss':>20}{'Accuracy':>20}{'Top5_Accuracy':>20}{'Score_diff':>20}    {'Normalized_score_diff':>20}{'N_samples':>14}")
    print('-------------------------------------------------------------------------------------------------------------------------------------------')
    print(f"{'Average':>15}{data_results['Loss']:>19.2f}{data_results['Accuracy']:>19.2f}{data_results['Top5_Accuracy']:>19.2f}{data_results['Score_diff']:>19.2f}    {data_results['Normalized_score_diff']:>19.2f}{num_samples:>20d}")
    print('===========================================================================================================================================')    
    

if __name__ == '__main__':
    '''
    # 3 Evaluation
    # Finally, we can evaluate the performance of the model. We first define appropriate environments. For benchmarking purposes, we include a trivial environment that merely runs SCIP.
    '''

    parser = argparse.ArgumentParser(description='Configuration for PyTorch model training.')

    # General settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--parent_model_dir', type=str, default='save_dir/branching_checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument("--parent_data_dir", type=str, default='save_dir/branching_data', help="Directory for parent data.")
    parser.add_argument('--parent_test_stats_dir', type=str, default='save_dir/branching_test_stats', help='Directory to save test stats.')
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps', help="Directory for instance data.")
    parser.add_argument("--parent_instances_metadata_dir", type=str, default='save_dir/instances/metadata', help="Directory for instance metadata.")

    parser.add_argument("--max_data_per_instance", type=int, default=20, help="Maximum number of data samples per instance.")
    parser.add_argument("--ntrain_instances", type=int, default=350, help="Number of training instances to collect data.")
    parser.add_argument("--nval_instances", type=int, default=50, help="Number of validation instances to collect data.")
    parser.add_argument("--ntest_instances", type=int, default=1000, help="Number of test instances to evaluate on.")
    parser.add_argument("--ntrain_data", type=int, default=float('inf'), help="Number of training data samples.")
    parser.add_argument("--nval_data", type=int, default=float('inf'), help="Number of training data samples.")
    parser.add_argument("--ntest_data", type=int, default=float('inf'), help="Number of training data samples.")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--select_option", type=str, default='first', choices=['first', 'random'], help="Option to select data samples from instances.")

    # model parameters for getting the correct model name
    parser.add_argument("--model_difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--model_code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--model_code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--model_code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--model_code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--model_ntrain_instances", type=int, default=350, help="Number of training instances to collect data.")
    parser.add_argument("--model_ntrain_data", type=int, default=float('inf'), help="Number of training instances to collect data.")
    parser.add_argument("--model_max_data_per_instance", type=int, default=20, help="Maximum number of data samples per instance.")
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

    parser.add_argument("--eval_split", type=str, default='test', choices=['train', 'test', 'val', 'all'], help="Split to evaluate.")
    parser.add_argument("--not_test_by_dir", action="store_true", help="Flag to determine whether to test by instance directories.")
    parser.add_argument("--load_model_dir", type=str, default='', help="Model directory to load.")
    parser.add_argument("--model_suffix", type=str, default='', help="Suffix to add to the model directory.")
    
    args = parser.parse_args()

    do_parallel = True
    seed = args.seed
    Ntrain_instances = args.ntrain_instances
    Nval_instances = args.nval_instances
    Ntest_instances = args.ntest_instances
    Ntrain_data = args.ntrain_data
    Nval_data = args.nval_data
    Ntest_data = args.ntest_data
    select_option = args.select_option
    max_data_per_instance = args.max_data_per_instance

    parent_model_dir = args.parent_model_dir
    parent_instances_dir = args.parent_instances_dir
    parent_instances_metadata_dir = args.parent_instances_metadata_dir
    parent_data_dir = args.parent_data_dir
    parent_test_stats_dir = args.parent_test_stats_dir
    # GNN parameters
    emb_size = args.emb_size
    edge_nfeats = args.edge_nfeats
    n_layers = args.n_layers
    test_by_dir = not args.not_test_by_dir
    test_batch_size = args.test_batch_size
    reevaluate = args.reevaluate

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We will first define pytorch geometric data classes to handle the bipartite graph data.
    # We can then prepare the data loaders.
    #### Load Train, Val, Test Data split
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

        if args.model_suffix != '':
            model_dir = model_dir + f'_{args.model_suffix}'
            modelname = modelname + f'_{args.model_suffix}'
    

    assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"

    load_checkpoint(policy, None, step='max', save_dir=model_dir, device=device)

    policy.eval()
    policy.to(device)
    
    
    parent_test_stats_dir = os.path.join(args.parent_test_stats_dir, modelname, 'by_dir')
    parent_test_stats_dir = os.path.join(parent_test_stats_dir, args.eval_split)
    os.makedirs(parent_test_stats_dir, exist_ok=True)  
    print_dash_str(f'[{args.eval_split}] Save stats to: {parent_test_stats_dir}')

    if args.eval_split == 'all':
        instances_dir_split = instances_dir_list
    else:
        train_instances_dir_split, val_instances_dir_split, test_instances_dir_split = get_instances_dir_train_val_test_split(instances_dir_split_file, instances_dir_list)
        instances_dir_split = test_instances_dir_split if args.eval_split == 'test' else val_instances_dir_split if args.eval_split == 'val' else train_instances_dir_split

    if len(instances_dir_split) == 0:
        print(f'No {args.eval_split} instance directories found in {instances_dir_split_file}. Please check the instances_dir_split_file.')
        exit()

    print(f'{args.eval_split}: {len(instances_dir_split)} to check ...')

    test_files_list = []

    for i_dir, instances_dir in enumerate(instances_dir_split):  #  tqdm.tqdm(instances_dir_split):            
        test_files = glob.glob(os.path.join(parent_data_dir, instances_dir, f'data_*.pkl.gz'))
        filtered_files = [file for file in test_files if int(os.path.basename(file).split('.')[0].split('_')[-1]) < 50]
        test_files = random.sample(filtered_files, min(500, len(filtered_files)))

        if len(test_files) == 0:
            print(f'No test files found for {instances_dir}. Skipping...')
            continue

        test_files_list.append(test_files)
        print(f'[{i_dir+1}/{len(instances_dir_split)}] {instances_dir}:: Number of test data points: {len(test_files)}.')
        if test_by_dir:
            print_dash_str(f'Evaluating {len(test_files)} test files for {instances_dir}. Save stats to {parent_test_stats_dir} ...')
            test_data = GraphDataset(test_files, edge_nfeats=edge_nfeats)
            test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=test_batch_size, num_workers=0, pin_memory=False, shuffle=False)
            stats_dir = os.path.join(parent_test_stats_dir, instances_dir)
            os.makedirs(stats_dir, exist_ok=True)
            stats_filename = os.path.join(stats_dir, 'eval_acc.json')
            if reevaluate or not os.path.exists(stats_filename):
                evaluate(policy, test_loader, device, stats_filename)
                print(f'Done evaluating all test files:: Save stats to {stats_filename} ...')
            load_and_summarize_performance(test_files, parent_data_dir, parent_test_stats_dir)

    test_files_all = [x for test_files in test_files_list for x in test_files]
    if test_by_dir:
        print_dash_str(f'Overall performance for {len(test_files_all)} test files from all instances directories...')
    else:
        print_dash_str(f'Evaluating {len(test_files_all)} test files from all instances directories...')
        test_data = GraphDataset(test_files_all, edge_nfeats=edge_nfeats)
        test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=test_batch_size, num_workers=0, pin_memory=False, shuffle=False)
        stats_dir = os.path.join(parent_test_stats_dir, 'all_instances')
        os.makedirs(stats_dir, exist_ok=True)
        stats_filename = os.path.join(stats_dir, 'eval_acc.json')
        if reevaluate or not os.path.exists(stats_filename):
            evaluate(policy, test_loader, device, stats_filename)
            print(f'Done evaluating all test files:: Save stats to {stats_filename} ...')
    
        
    print_dash_str(f'Individual class performances')
    for instances_dir, test_files in zip(instances_dir_list, test_files_list):
        print_dash_str(f'Performance for {instances_dir}')
        load_and_summarize_performance(test_files, parent_data_dir, parent_test_stats_dir)

    print_dash_str(f'Overall performance for all instances directories...')
    load_and_summarize_performance(test_files_all, parent_data_dir, parent_test_stats_dir)
