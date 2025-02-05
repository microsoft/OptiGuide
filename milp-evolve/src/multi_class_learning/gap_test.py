import os
import torch
import argparse
import pdb
import numpy as np

from gap_model import MyGNNAttn
from gap_dataset import getDataloaders
from utils import load_gzip, load_json, save_json, load_checkpoint, set_seed, get_test_paths, get_test_paths_by_dir, print_dash_str
from utils import get_data_train_val_test_split, get_instances_dir_train_val_test_split, get_model_name_from_base_name, get_gzip_files_by_extension
from scipy.stats import pearsonr
import tqdm

path_to_randomness_control_set = 'SCIP_settings/randomness_control.set'

def evaluate(model, dataloader, device, dtype, criterion, parent_test_stats_dir, 
             label_clip_lb=-float('inf'), label_clip_ub=float('inf'), label_mult=1, 
             do_log_transform=False, name='test'):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_deviation = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    os.makedirs(parent_test_stats_dir, exist_ok=True)

    with torch.no_grad():  # No gradients needed
        for batch in tqdm.tqdm(dataloader):
            batch = batch.to(device)
            labels = batch.label.to(device, dtype).reshape(-1, 1).clip(label_clip_lb, label_clip_ub) * label_mult
            preds = model(batch)

            preds = preds.clip(label_clip_lb * label_mult, label_clip_ub * label_mult)

            if do_log_transform:
                preds = torch.log1p(preds + 1)
                labels = torch.log1p(labels + 1)
            
            loss = criterion(preds, labels)
            total_loss += loss.item() * labels.size(0)
            total_deviation += (torch.abs(preds.reshape(-1) - labels.reshape(-1))).sum().item()
            preds_cpu = preds.reshape(-1).to(torch.float32).cpu().numpy()
            labels_cpu = labels.reshape(-1).to(torch.float32).cpu().numpy()
            total_samples += labels.size(0)

            all_preds += preds_cpu.tolist()
            all_labels += labels_cpu.tolist()

    average_loss = total_loss / total_samples
    average_deviation = total_deviation / total_samples

    try:
        corr = np.corrcoef(all_preds, all_labels)[0, 1]
    except Exception as e:
        print("Fail to calculate correlation coefficient.", e)
        corr = 0
    print(f'{name}:: R: {corr:.2f}, Average Loss: {average_loss:.2f}, Average Deviation: {average_deviation:.2f}, Total Samples: {total_samples}')
    if len(all_labels) >= 2:
        correlation, _ = pearsonr(all_preds, all_labels)
        print(f'Correlation: {correlation:.2f}')

    save_json(os.path.join(parent_test_stats_dir, 'preds_labels.json'), {'preds': all_preds, 'labels': all_labels})
    print('Test statistics saved to', os.path.join(parent_test_stats_dir, 'preds_labels.json'))


    return average_loss, average_deviation, total_samples, all_preds, all_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for PyTorch model training.')

    # General settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument("--parent_model_dir", type=str, default='save_dir/gap_checkpoints', help="Directory to save model checkpoints.")
    parser.add_argument('--parent_test_stats_dir', type=str, default='save_dir/gap_test_stats', help='Directory to save test stats.')
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps', help="Directory for instance data.")
    parser.add_argument("--parent_instances_metadata_dir", type=str, default='save_dir/instances/metadata', help="Directory for instance metadata.")
    parser.add_argument("--parent_data_dir", type=str, default='save_dir/gap_data', help="Directory for parent data.")
    
    parser.add_argument("--label_mult", type=float, default=100, help="Label multiplier.")
    parser.add_argument("--label_clip_lb", type=float, default=-float('inf'), help="Label clipping lower bound.")
    parser.add_argument("--label_clip_ub", type=float, default=float('inf'), help="Label clipping upper bound.")
    parser.add_argument("--label_key", type=str, default='lp_ip_gap', choices=['lp_ip_gap', 'lp_value', 'dual_value', 'obj_value', 'solve_time', 'n_nodes'])

    ### not used
    parser.add_argument("--ntrain_instances", type=int, default=0, help="Number of training instances to collect data.")
    parser.add_argument("--nval_instances", type=int, default=0, help="Number of validation instances to collect data.")
    parser.add_argument("--ntest_instances", type=int, default=100, help="Number of testing instances to collect data.")
    parser.add_argument("--select_option", type=str, default='first', choices=['first', 'random'], help="Option to select data samples from instances.")
    ###
    
    parser.add_argument("--test_batch_size", type=int, default=128, help="Test batch size.")
    parser.add_argument("--test_instances_dir_split_file", type=str, default='', help="File containing the instance directories to split.")
    parser.add_argument("--reevaluate", action="store_true", help="Flag to reevaluate instances.")

    # GNN arguments
    parser.add_argument("--emb_size", type=int, default=64, help="Embedding size of the GNN.")
    parser.add_argument("--edge_nfeats", type=int, default=1, help="Number of edge features.", choices=[1, 2])
    parser.add_argument("--n_layers", type=int, default=1, help="Number of GNN layers.")

    parser.add_argument("--do_log_transform", action="store_true", help="Flag to determine whether to log transform the labels.")

    # model parameters
    parser.add_argument("--model_difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--model_code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--model_code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--model_ntrain_instances", type=int, default=350, help="Number of training instances to collect data.")
    parser.add_argument("--model_code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--model_code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--model_load_data_split", action="store_true", help="Flag to determine whether to load data split.")
    parser.add_argument("--model_data_split_file", type=str, default='', help="File containing the data split.")
    parser.add_argument("--model_code_str", type=str, default='code', help="String to identify the code instances.")
    parser.add_argument("--data_split_file", type=str, default='', help="File containing the data split.")
    
    ### not used
    parser.add_argument("--difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")
    ####
    
    parser.add_argument("--eval_split", type=str, default='test', choices=['train', 'test', 'val'], help="Split to evaluate.")
    parser.add_argument("--load_model_dir", type=str, default='', help="Model directory to load.")
    parser.add_argument("--model_suffix", type=str, default='', help="Suffix to add to the model directory.")
    parser.add_argument("--load_step", type=int, default=-1, help="Step to load the model.")
    parser.add_argument("--max_token_attn", type=int, default=512, help="Maximum number of tokens for attention layers.")
    args = parser.parse_args()

    # if args.model_difficulty == '':
    #     for arg in ['difficulty', 'code_start_idx', 'code_end_idx', 'code_exclude_idxs', 'code_idx_difficulty_list',
    #                 'ntrain_instances', 'data_split_file', 'code_str']:
    #         if hasattr(args, arg):
    #             setattr(args, f'model_{arg}', getattr(args, arg))
        
    seed = args.seed

    label_clip_lb = args.label_clip_lb
    label_clip_ub = args.label_clip_ub
    label_mult = args.label_mult
    label_key = args.label_key

    parent_model_dir = args.parent_model_dir
    parent_instances_dir = args.parent_instances_dir
    parent_data_dir = args.parent_data_dir
    parent_instances_metadata_dir = args.parent_instances_metadata_dir
    
    test_batch_size = args.test_batch_size
    # GNN parameters
    emb_size = args.emb_size
    edge_nfeats = args.edge_nfeats
    n_layers = args.n_layers
    do_log_transform = args.do_log_transform

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = MyGNNAttn(emb_size=emb_size, edge_nfeats=edge_nfeats, n_gnn_iters=n_layers, max_token_attn=args.max_token_attn)

    criterion = torch.nn.HuberLoss() # torch.nn.BCEWithLogitsLoss()

    # load model name
    if args.model_load_data_split:
        basename = os.path.basename(os.path.dirname(args.model_data_split_file)) + '_' + os.path.basename(args.model_data_split_file).replace('.json', '')
        modelname = get_model_name_from_base_name(basename, args, parser, mode='test')
        if 'in_class' in os.path.basename(args.model_data_split_file):
            modelname += '_in_class'
        model_dir = os.path.join(parent_model_dir, modelname)
    else:
        basename = os.path.basename(os.path.dirname(args.data_split_file))
        modelname = get_model_name_from_base_name(basename, args, parser, mode='test')
        model_dir = os.path.join(parent_model_dir, modelname)
   
    if args.load_model_dir != '':
        model_dir = args.load_model_dir 
        modelname = os.path.basename(model_dir)
    else:
        modelname = os.path.basename(model_dir)
        model_dir = model_dir + '_attn' + (f'-max{args.max_token_attn}' if args.max_token_attn != parser.get_default('max_token_attn') else '')
        modelname = modelname + '_attn' + (f'-max{args.max_token_attn}' if args.max_token_attn != parser.get_default('max_token_attn') else '')

        if args.model_suffix:
            model_dir = model_dir + f'_{args.model_suffix}'
            modelname = modelname + f'_{args.model_suffix}'


    assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist."
    load_checkpoint(model, None, step='max' if args.load_step < 0 else args.load_step, save_dir=model_dir, device=device)
    model.to(device, dtype)

    #### Load Train, Val, Test Data split
    data_split = load_json(args.data_split_file)
    test_split = data_split[args.eval_split]
    print_dash_str('Testing ...')
    
    parent_test_stats_dir = os.path.join(args.parent_test_stats_dir, modelname)
    data_parent_test_stats_dir = os.path.join(parent_test_stats_dir, os.path.basename(os.path.dirname(args.data_split_file)))

    test_dataloader = getDataloaders(test_split, batch_size=test_batch_size, label_key=label_key)
    

    average_loss, average_deviation, total_samples_all, all_preds, all_labels = evaluate(model, test_dataloader, device, dtype, criterion, data_parent_test_stats_dir, 
                                                        label_clip_lb=label_clip_lb, label_clip_ub=label_clip_ub,
                                                        label_mult=label_mult, do_log_transform=do_log_transform,
                                                        name=args.eval_split)

    print(f'Test {args.data_split_file}:: average loss {average_loss:.2f}, average deviation {average_deviation:.2f}, total samples {total_samples_all}.')
        

    # print mean preds performance: mean of the labels
    all_mean_preds = [np.mean(all_labels)] * len(all_labels)
    mean_loss_all = criterion(torch.tensor(all_mean_preds), torch.tensor(all_labels)).item()
    mean_deviation_all = sum([abs(pred - label) for pred, label in zip(all_mean_preds, all_labels)]) / len(all_mean_preds)
    mean_correlation_all, _ = pearsonr(all_mean_preds, all_labels)
    print_dash_str(f'Mean Preds [{np.mean(all_labels):.2f}]:: Average Loss: {mean_loss_all:.2f}, Average Deviation: {mean_deviation_all:.2f}, R: {mean_correlation_all:.2f}. Total Samples: {total_samples_all}')

    all_median_preds = [np.median(all_labels)] * len(all_labels)
    median_loss_all = criterion(torch.tensor(all_median_preds), torch.tensor(all_labels)).item()
    median_deviation_all = sum([abs(pred - label) for pred, label in zip(all_median_preds, all_labels)]) / len(all_median_preds)
    median_correlation_all, _ = pearsonr(all_median_preds, all_labels)
    print_dash_str(f'Median Preds [{np.median(all_labels):.2f}]:: Average Loss: {median_loss_all:.2f}, Average Deviation: {median_deviation_all:.2f}, R: {median_correlation_all:.2f}. Total Samples: {total_samples_all}')

