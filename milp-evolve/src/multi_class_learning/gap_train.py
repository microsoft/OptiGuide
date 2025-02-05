import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pyscipopt as pyopt
import torch
import tqdm
from gap_model import MyGNNAttn
from gap_context import getContext
from gap_dataset import getDataloaders
from torch.utils.tensorboard import SummaryWriter
from utils import (get_model_name_from_base_name, load_checkpoint, load_json, 
                   print_dash_str, save_checkpoint, save_json, set_seed)


def train(train_dataloader, model, criterion, optimizer, scheduler, device, dtype, num_epochs, 
          label_clip_lb=-float('inf'), label_clip_ub=float('inf'), label_mult=1, 
          val_dataloader=None, test_dataloader=None, eval_every=1000, save_every=1000, model_dir='save_dir/checkpoints',
          do_log_transform=False, log_dir='save_dir/gap_logs'):
    model.train()

    num_gradient_steps = 0
    writer = SummaryWriter(log_dir)
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_deviation = 0
        total_samples = 0

        for batch in train_dataloader:
            if num_gradient_steps % eval_every == 0 and val_dataloader is not None:
                print_dash_str(f'Evaluating at epoch {epoch+1}, num_gradient_steps {num_gradient_steps}')
                evaluate(model, val_dataloader, device, dtype, criterion, writer, num_gradient_steps, 
                         label_clip_lb=label_clip_lb, label_clip_ub=label_clip_ub, label_mult=label_mult,
                         do_log_transform=do_log_transform)
                print_dash_str()

            if num_gradient_steps % save_every == 0:
                save_checkpoint(model, num_gradient_steps, optimizer, model_dir)

            batch = batch.to(device, dtype)
            labels = batch.label.to(device, dtype).reshape(-1, 1).clip(label_clip_lb, label_clip_ub) * label_mult
            optimizer.zero_grad()
            preds = model(batch)

            if do_log_transform:
                preds = torch.log1p(preds + 1)
                labels = torch.log1p(labels + 1)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            num_gradient_steps += 1

            running_loss += loss.item()
            mean_deviation = (torch.abs(preds.reshape(-1) - labels.reshape(-1))).mean().item()
            total_deviation += mean_deviation * labels.size(0)
            total_samples += labels.size(0)

            writer.add_scalar('Loss/train', loss.item(), num_gradient_steps)
            writer.add_scalar('Deviation/train', mean_deviation, num_gradient_steps)

        if scheduler is not None:
            scheduler.step()
        
        torch.cuda.empty_cache()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader):.2f}. Average Deviation: {total_deviation/total_samples:.2f} ({num_gradient_steps} gradient steps so far).')
        writer.add_scalar('Loss/train_epoch', running_loss/len(train_dataloader), epoch)
        writer.add_scalar('Deviation/train_epoch', total_deviation/total_samples, epoch)

    save_checkpoint(model, num_gradient_steps, optimizer, model_dir)
    writer.close()

    if test_dataloader:
        evaluate(model, test_dataloader, device, dtype, criterion, writer, num_gradient_steps, 
                label_clip_lb=label_clip_lb, label_clip_ub=label_clip_ub, label_mult=label_mult,
                do_log_transform=do_log_transform, name="test")
    

def evaluate(model, dataloader, device, dtype, criterion, writer, num_gradient_steps, 
             label_clip_lb=-float('inf'), label_clip_ub=float('inf'), label_mult=1,
             do_log_transform=False, name="val"):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_deviation = 0
    total_samples = 0

    preds_all = []
    labels_all = []
    with torch.no_grad():  # No gradients needed
        for batch in tqdm.tqdm(dataloader):
            batch = batch.to(device, dtype)
            labels = batch.label.to(device, dtype).reshape(-1, 1).clip(label_clip_lb, label_clip_ub) * label_mult
            preds = model(batch)
            loss = criterion(preds, labels)
            if do_log_transform:
                preds = torch.log1p(preds + 1)
                labels = torch.log1p(labels + 1)

            preds = preds.clip(label_clip_lb * label_mult, label_clip_ub * label_mult)

            total_loss += loss.item()
            total_deviation += (torch.abs(preds.reshape(-1) - labels.reshape(-1))).sum().item()
         
            preds_cpu = preds.reshape(-1).to(torch.float32).cpu().numpy()
            labels_cpu = labels.reshape(-1).to(torch.float32).cpu().numpy()
            preds_all += preds_cpu.reshape(-1).tolist()
            labels_all += labels_cpu.reshape(-1).tolist()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)
    average_deviation = total_deviation / total_samples
    try:
        corr = np.corrcoef(preds_all, labels_all)[0, 1]
    except Exception as e:
        print("Fail to calculate correlation coefficient.", e)
        corr = 0
    print(f'{name}:: R: {corr:.2f}, Average Loss: {average_loss:.2f}, Average Deviation: {average_deviation:.2f}, Total Samples: {total_samples}')
    writer.add_scalar(f'Loss/{name}', average_loss, num_gradient_steps)
    writer.add_scalar(f'Deviation/{name}', average_deviation, num_gradient_steps)

    return average_loss, average_deviation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure training settings for the ML model.")

    # Integer arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--parent_model_dir", type=str, default='save_dir/gap_checkpoints', help="Directory to save model checkpoints.")
    parser.add_argument("--parent_log_dir", type=str, default='save_dir/gap_logs', help="Directory to save logs.")
    parser.add_argument("--parent_data_dir", type=str, default='save_dir/gap_data', help="Directory for parent data.")
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps', help="Directory for instance data.")
    parser.add_argument("--parent_instances_metadata_dir", type=str, default='save_dir/instances/metadata', help="Directory for instance metadata.")
    parser.add_argument("--label_mult", type=float, default=100, help="Label multiplier.")
    parser.add_argument("--label_clip_lb", type=float, default=-float('inf'), help="Label clipping lower bound.")
    parser.add_argument("--label_clip_ub", type=float, default=float('inf'), help="Label clipping upper bound.")
    parser.add_argument("--label_key", type=str, default='lp_ip_gap', choices=['lp_ip_gap', 'lp_value', 'dual_value', 'obj_value', 'solve_time', 'n_nodes'])
  
    parser.add_argument("--nb_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluation frequency.")
    parser.add_argument("--save_every", type=int, default=100, help="Model saving frequency.")  
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=128, help="Validation batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--lr_decay", type=float, default=0.1, help="Learning rate.")

    # GNN arguments
    parser.add_argument("--emb_size", type=int, default=64, help="Embedding size of the GNN.")
    parser.add_argument("--edge_nfeats", type=int, default=1, help="Number of edge features.", choices=[1, 2])
    parser.add_argument("--n_layers", type=int, default=1, help="Number of GNN layers.")

    parser.add_argument("--do_log_transform", action="store_true", help="Flag to determine whether to log transform the labels.")

    # Data arguments to decide how to split the data from the instance split file
    parser.add_argument("--ntrain_instances", type=int, default=350, help="Number of training instances to collect data.")
    parser.add_argument("--nval_instances", type=int, default=50, help="Number of validation instances to collect data.")
    parser.add_argument("--ntest_instances", type=int, default=100, help="Number of testing instances to collect data.")
    parser.add_argument("--select_option", type=str, default='first', choices=['first', 'random'], help="Option to select data samples from instances.")

    parser.add_argument("--data_split_file", type=str, default='', help="File containing the data split.")

    ### not used
    parser.add_argument("--difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")
    ###
    
    parser.add_argument("--resume", action="store_true", help="Flag to determine whether to resume training.")
    parser.add_argument("--load_model_dir", type=str, default='', help="Model directory to resume training.")
    parser.add_argument("--model_suffix", type=str, default='', help="Suffix to add to the model directory.")
    parser.add_argument("--max_token_attn", type=int, default=512, help="Maximum number of tokens for attention layers.")
    parser.add_argument("--exclude_keys", type=str, nargs='*', default=[], help="Keys to exclude from the model state_dict (if resume).")
    args = parser.parse_args()


    ## save dir
    lr = args.learning_rate
    lr_decay = args.lr_decay
    num_epochs = args.nb_epochs
    eval_every = args.eval_every
    save_every = args.save_every
    label_clip_lb = args.label_clip_lb
    label_clip_ub = args.label_clip_ub
    label_mult = args.label_mult
    label_key = args.label_key

    Ntrain_instances = args.ntrain_instances
    Nval_instances = args.nval_instances
    Ntest_instances = args.ntest_instances
    select_option = args.select_option

    seed = args.seed
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    parent_data_dir = args.parent_data_dir
    parent_instances_dir = args.parent_instances_dir
    parent_instances_metadata_dir = args.parent_instances_metadata_dir

    # GNN parameters
    emb_size = args.emb_size
    edge_nfeats = args.edge_nfeats
    n_layers = args.n_layers

    do_log_transform = args.do_log_transform

    basename = os.path.basename(os.path.dirname(args.data_split_file)) + '_' + os.path.basename(args.data_split_file).replace('.json', '')
    modelname = get_model_name_from_base_name(basename, args, parser, mode='train')
    model_dir = os.path.join(args.parent_model_dir, modelname)
    log_dir = os.path.join(args.parent_log_dir, modelname)

    model_dir = model_dir + '_attn' + (f'-max{args.max_token_attn}' if args.max_token_attn != parser.get_default('max_token_attn') else '')
    log_dir = log_dir + '_attn' + (f'-max{args.max_token_attn}' if args.max_token_attn != parser.get_default('max_token_attn') else '')

    if args.model_suffix:
        model_dir = model_dir + f'_{args.model_suffix}'
        log_dir = log_dir + f'_{args.model_suffix}'

    if args.resume:
        load_model_dir = model_dir if (args.load_model_dir == '' or not os.path.exists(args.load_model_dir)) else args.load_model_dir
        load_modelname = os.path.basename(load_model_dir)
        model_dir = model_dir + f'_resume_from_{load_modelname}'
        log_dir = log_dir + f'_resume_from_{load_modelname}'    

    set_seed(seed)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Device: {device}, data type: {dtype}, Model dir: {model_dir}, Log dir: {log_dir}.")

    # Save args in model_dir as a json file
    save_json(os.path.join(model_dir, 'train_args.json'), vars(args))

    #### Load Train, Val, Test Data split
    data_split = load_json(args.data_split_file)
    train_split, val_split, test_split = data_split['train'], data_split['val'], data_split['test']
        
    
    train_split = [x for x in train_split] 
    val_split = [x for x in val_split]
    test_split = [x for x in test_split]

    train_dataloader = getDataloaders(train_split, label_key=label_key)
    val_dataloader = getDataloaders(val_split, label_key=label_key)
    test_dataloader = getDataloaders(test_split, label_key=label_key)

    print_dash_str()
    print_dash_str(f'Total # Train data = {len(train_split)} # val data = {len(val_split)}, # test data = {len(test_split)}.')
    ### Training
    print_dash_str('Training ...')
    model = MyGNNAttn(emb_size=emb_size, edge_nfeats=edge_nfeats, n_gnn_iters=n_layers, max_token_attn=args.max_token_attn)
   
    model.to(device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion = torch.nn.HuberLoss() # torch.nn.BCEWithLogitsLoss()
    scheduler = None # torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay) 
       
    if args.resume:
        load_checkpoint(model, None, step='max', save_dir=load_model_dir, device=device, exclude_keys=args.exclude_keys)
        print(f'Load Model Checkpoint from {load_model_dir}')     

    
    train(train_dataloader, model, criterion, optimizer, scheduler, device, dtype, num_epochs, 
          label_clip_lb=label_clip_lb, label_clip_ub=label_clip_ub, 
          label_mult=label_mult, val_dataloader=val_dataloader, test_dataloader=test_dataloader, eval_every=eval_every, 
          save_every=save_every, model_dir=model_dir, log_dir=log_dir, 
          do_log_transform=do_log_transform)
    
    