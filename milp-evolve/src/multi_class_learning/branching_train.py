import os
import pdb
import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch_geometric
from branching_dataset import GraphDataset
from branching_model import GNNPolicy
from utils import set_seed, save_checkpoint, load_checkpoint, get_train_paths, print_dash_str
from utils import get_data_train_val_test_split_list_by_dir, load_json
from torch.utils.tensorboard import SummaryWriter


def train(policy, optimizer, train_dataloader, device, epochs, start_step=0,
          score_th=float('inf'), eval_every=1000, save_every=1000, print_every=50, 
          model_dir='save_dir/branching_checkpoints', log_dir='save_dir/branching_logs', 
          val_dataloader=None, loss_option='classification'):
    policy.train()
    writer = SummaryWriter(log_dir=log_dir)
    
    num_gradient_steps = start_step
    for epoch in range(epochs):
        mean_loss = 0
        mean_acc = 0
        mean_top5_acc = 0
        mean_score_diff = 0
        mean_normalized_score_diff = 0

        n_samples_processed = 0
        for batch in train_dataloader:
            if val_dataloader is not None and num_gradient_steps % eval_every == 0:
                print_dash_str(f'Evaluating at epoch {epoch+1}, num_gradient_steps {num_gradient_steps}')
                valid_loss, valid_acc, valid_top5_acc, mean_score_diff, mean_normalized_score_diff = evaluate(policy, val_dataloader, device, writer, num_gradient_steps)
                print_dash_str(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}, top 5 accuracy {valid_top5_acc:0.3f} score difference [abs] {mean_score_diff:.3f} [relative] {mean_normalized_score_diff:.3f}")
            
            if num_gradient_steps % save_every == 0:
                save_checkpoint(policy, num_gradient_steps, optimizer, model_dir)
            
            # print loss
            if num_gradient_steps % print_every == 0:
                if n_samples_processed > 0:
                    print(f"Gradient Step {num_gradient_steps}: Train loss: {mean_loss / n_samples_processed:0.3f}, accuracy {mean_acc / n_samples_processed:0.3f}, "
                            f"Top 5 accuracy {mean_top5_acc / n_samples_processed:0.3f}, [absolute] {mean_score_diff / n_samples_processed:.3f} "
                            f"[relative] {mean_normalized_score_diff / n_samples_processed:.3f}")

            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )

            if score_th < float('inf'):
                select_indices = batch.candidate_scores.max(axis=-1).values < score_th
                logits = logits[select_indices]
                batch = batch[select_indices]
                if len(logits) == 0:
                    continue
            else:
                # Index the results by the candidates, and split and pad them
                logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)

            if loss_option == 'classification':
                # Compute the usual cross-entropy classification loss
                loss = F.cross_entropy(logits, batch.candidate_choices)
            else:
                # regression: regress with the scores
                loss = F.mse_loss(logits, batch.candidate_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_gradient_steps += 1

            # calculate train statistics
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()
            top5_acc = (true_scores.gather(-1, logits.topk(min(5, logits.size(-1))).indices) == true_bestscore).float().max(dim=-1).values.mean().item()
            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            mean_top5_acc += top5_acc * batch.num_graphs
            n_samples_processed += batch.num_graphs

            # torch.save(policy.state_dict(), "trained_params.pkl")
            writer.add_scalar("Loss/train", loss.item(), num_gradient_steps)
            writer.add_scalar("Accuracy/train", accuracy, num_gradient_steps)
            writer.add_scalar("Top5_Accuracy/train", top5_acc, num_gradient_steps)
            # New stats
            score_diff = (true_bestscore - true_scores.gather(-1, predicted_bestindex).clip(0)).mean().item()
            normalized_score_diff = ((true_bestscore - true_scores.gather(-1, predicted_bestindex).clip(0)) / true_bestscore).mean().item()
            mean_score_diff += score_diff * batch.num_graphs
            mean_normalized_score_diff += normalized_score_diff * batch.num_graphs

            writer.add_scalar("Score_diff/train", score_diff, num_gradient_steps)
            writer.add_scalar("Normalized_score_diff/train", normalized_score_diff, num_gradient_steps)


        mean_loss /= n_samples_processed
        mean_acc /= n_samples_processed
        mean_top5_acc /= n_samples_processed
        mean_score_diff /= n_samples_processed
        mean_normalized_score_diff /= n_samples_processed
        print(f"Epoch {epoch+1}: Train loss: {mean_loss:0.3f}, accuracy {mean_acc:0.3f}, top 5 accuracy {mean_top5_acc:0.3f}, "
              f"[absolute] {mean_score_diff:.3f} [relative] {mean_normalized_score_diff:.3f}")
        
        writer.add_scalar("Loss/Epoch_train", mean_loss, epoch)
        writer.add_scalar("Accuracy/Epoch_train", mean_acc, epoch)
        writer.add_scalar("Top5_Accuracy/Epoch_train", mean_top5_acc, epoch)
        writer.add_scalar("Score_diff/Epoch_train", mean_score_diff, epoch)
        writer.add_scalar("Normalized_score_diff/Epoch_train", mean_normalized_score_diff, epoch)


    save_checkpoint(policy, num_gradient_steps, optimizer, model_dir)
    # torch.save(policy.state_dict(), "trained_params.pkl")
    writer.close()
    

def evaluate(policy, data_loader, device, writer, num_gradient_steps):
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
            normalized_score_diff = ((true_bestscore - true_scores.gather(-1, predicted_bestindex)) / true_bestscore).mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            mean_top5_acc += top5_acc * batch.num_graphs

            mean_score_diff += score_diff * batch.num_graphs
            mean_normalized_score_diff += normalized_score_diff * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    mean_top5_acc /= n_samples_processed
    mean_score_diff /= n_samples_processed
    mean_normalized_score_diff /= n_samples_processed
    writer.add_scalar("Loss/val", mean_loss, num_gradient_steps)
    writer.add_scalar("Accuracy/val", mean_acc, num_gradient_steps)
    writer.add_scalar("Top5_Accuracy/val", mean_top5_acc, num_gradient_steps)
    writer.add_scalar("Score_diff/val", mean_score_diff, num_gradient_steps)
    writer.add_scalar("Normalized_score_diff/val", mean_normalized_score_diff, num_gradient_steps)

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure training settings for the ML model.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--parent_model_dir", type=str, default='save_dir/branching_checkpoints', help="Directory to save model checkpoints.")
    parser.add_argument("--parent_log_dir", type=str, default='save_dir/branching_logs', help="Directory to save logs.")
    parser.add_argument("--parent_data_dir", type=str, default='save_dir/branching_data', help="Directory for parent data.")
    parser.add_argument("--parent_instances_dir", type=str, default='save_dir/instances/mps', help="Directory for instance data.")
    # parser.add_argument("--parent_data_metadata_dir", type=str, default='save_dir/branching_data/metadata', help="Directory for data metadata.")
    parser.add_argument("--parent_instances_metadata_dir", type=str, default='save_dir/instances/metadata', help="Directory for instance metadata.")
    
    parser.add_argument("--nb_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluation frequency.")
    parser.add_argument("--save_every", type=int, default=100, help="Model saving frequency.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Validation batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")

    # GNN arguments
    parser.add_argument("--emb_size", type=int, default=64, help="Embedding size of the GNN.")
    parser.add_argument("--edge_nfeats", type=int, default=1, help="Number of edge features.", choices=[1, 2])
    parser.add_argument("--n_layers", type=int, default=1, help="Number of GNN layers.")

    # Data arguments to decide how to split the data from the instance split file
    parser.add_argument("--max_data_per_instance", type=int, default=50, help="Maximum number of data samples per instance.")
    parser.add_argument("--ntrain_instances", type=int, default=35, help="Number of training instances to collect data.")
    parser.add_argument("--nval_instances", type=int, default=5, help="Number of validation instances to collect data.")
    parser.add_argument("--ntest_instances", type=int, default=10, help="Number of testing instances to collect data.")
    parser.add_argument("--ntrain_data", type=int, default=float('inf'), help="Number of training data samples.")
    parser.add_argument("--nval_data", type=int, default=float('inf'), help="Number of training data samples.")
    parser.add_argument("--ntest_data", type=int, default=float('inf'), help="Number of training data samples.")

    parser.add_argument("--ntrain_instances_dir_train", type=int, default=-1, help="Number of training instances to collect data.")
    parser.add_argument("--nval_instances_dir_train", type=int, default=-1, help="Number of validation instances to collect data.")
    parser.add_argument("--ntest_instances_dir_train", type=int, default=-1, help="Number of testing instances to collect data.")
    parser.add_argument("--ntrain_instances_dir_val", type=int, default=-1, help="Number of training data samples.")
    parser.add_argument("--nval_instances_dir_val", type=int, default=-1, help="Number of training data samples.")
    parser.add_argument("--ntest_instances_dir_val", type=int, default=-1, help="Number of training data samples.")
    
    parser.add_argument("--select_option", type=str, default='first', choices=['first', 'random'], help="Option to select data samples from instances.")

    parser.add_argument("--instances_dir_split_file", type=str, default='', help="File containing the instance directories to split.")
    
    parser.add_argument("--difficulty", type=str, nargs='*', default=['easy', 'medium'], help="Difficulty of instances to process.")
    parser.add_argument("--code_start_idx", type=int, default=0, help="Starting index for code instances.")
    parser.add_argument("--code_end_idx", type=int, default=8, help="Ending index for code instances.")
    parser.add_argument("--code_exclude_idxs", type=int, nargs="*", default=[], help="Indices to exclude for code instances.")
    parser.add_argument("--code_idx_difficulty_list", type=tuple, nargs="*", default=[], help="List of tuples of code instance index and difficulty.")
    parser.add_argument("--code_str", type=str, default='code', help="String to identify the code instances.")
    
    parser.add_argument("--resume", action="store_true", help="Flag to determine whether to resume training.")
    parser.add_argument("--load_model_dir", type=str, default='', help="Model directory to resume training.")
    parser.add_argument("--model_suffix", type=str, default='', help="Suffix to add to the model directory.")
    args = parser.parse_args()

    learning_rate = args.learning_rate
    nb_epochs = args.nb_epochs
    eval_every = args.eval_every
    save_every = args.save_every
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    # data related parameters
    max_data_per_instance = args.max_data_per_instance
    Ntrain_instances = args.ntrain_instances
    Nval_instances = args.nval_instances
    Ntest_instances = args.ntest_instances
    Ntrain_data = args.ntrain_data
    Nval_data = args.nval_data
    Ntest_data = args.ntest_data
    select_option = args.select_option

    Ntrain_instances_dir_train = args.ntrain_instances_dir_train
    Nval_instances_dir_train = args.nval_instances_dir_train
    Ntest_instances_dir_train = args.ntest_instances_dir_train
    Ntrain_instances_dir_val = args.ntrain_instances_dir_val
    Nval_instances_dir_val = args.nval_instances_dir_val
    Ntest_instances_dir_val = args.ntest_instances_dir_val

    # directories
    parent_model_dir = args.parent_model_dir
    parent_log_dir = args.parent_log_dir
    parent_data_dir = args.parent_data_dir
    parent_instances_dir = args.parent_instances_dir
    parent_instances_metadata_dir = args.parent_instances_metadata_dir

    # GNN parameters
    emb_size = args.emb_size
    edge_nfeats = args.edge_nfeats
    n_layers = args.n_layers

    instances_dir_list, instances_split_files, model_dir, log_dir = get_train_paths(args, parser)
    
    assert os.path.exists(args.instances_dir_split_file), f'Instances dir split file {args.instances_dir_split_file} does not exist.'
    instances_dir_split_file = args.instances_dir_split_file
    print(f'Loading instances_dir_split_file from {instances_dir_split_file}...')
    dir_file = load_json(instances_dir_split_file)
    instances_dir_list = dir_file['train'] + dir_file['val'] + dir_file['test']


    if args.model_suffix:
        model_dir = model_dir + f'_{args.model_suffix}'
        log_dir = log_dir + f'_{args.model_suffix}'

    if args.resume:
        load_model_dir = model_dir if (args.resume_model_dir == '' or not os.path.exists(args.resume_model_dir)) else args.resume_model_dir
        load_modelname = os.path.basename(load_model_dir)
        model_dir = model_dir + f'_resume_from_{load_modelname}'
        log_dir = log_dir + f'_resume_from_{load_modelname}'
    

    print(f'Model is saved to {model_dir}, logs are saved to {log_dir}.')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    print('device', device)

    policy = GNNPolicy(emb_size=emb_size, edge_nfeats=edge_nfeats, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)


    #### Load Train, Val, Test Data split
    ########################## load train split ##########################
    if Ntrain_instances_dir_train >= 0 and Nval_instances_dir_train >= 0 and Ntest_instances_dir_train >= 0:
        Ntrain_instances_, Nval_instances_, Ntest_instances_ = Ntrain_instances_dir_train, Nval_instances_dir_train, Ntest_instances_dir_train
    else:
        Ntrain_instances_, Nval_instances_, Ntest_instances_ = Ntrain_instances, Nval_instances, Ntest_instances
    train_split, _, _ = get_data_train_val_test_split_list_by_dir(instances_dir_split_file, parent_instances_dir, 
                                                                  parent_instances_metadata_dir, parent_data_dir, instances_dir_list, 
                                                                  task='branching', select_option=select_option, 
                                                                  Ntrain_instances=Ntrain_instances_, 
                                                                  Nval_instances=Nval_instances_, Ntest_instances=Ntest_instances_, 
                                                                  Ntrain_data=Ntrain_data, Nval_data=Nval_data, Ntest_data=Ntest_data, 
                                                                  max_data_per_instance=max_data_per_instance,
                                                                  select_splits=['train']) 
    ########################## load val split ##########################
    if Ntrain_instances_dir_val >= 0 and Nval_instances_dir_val >= 0 and Ntest_instances_dir_val >= 0:
        Ntrain_instances_, Nval_instances_, Ntest_instances_ = Ntrain_instances_dir_val, Nval_instances_dir_val, Ntest_instances_dir_val
    else:
        Ntrain_instances_, Nval_instances_, Ntest_instances_ = Ntrain_instances, Nval_instances, Ntest_instances
    _, val_split, _ = get_data_train_val_test_split_list_by_dir(instances_dir_split_file, parent_instances_dir, 
                                                                parent_instances_metadata_dir, parent_data_dir, instances_dir_list, 
                                                                task='branching', select_option=select_option, 
                                                                Ntrain_instances=Ntrain_instances_, 
                                                                Nval_instances=Nval_instances_, Ntest_instances=Ntest_instances_, 
                                                                Ntrain_data=Ntrain_data, Nval_data=Nval_data, Ntest_data=Ntest_data, 
                                                                max_data_per_instance=max_data_per_instance,
                                                                select_splits=['val']) 

    print_dash_str(f'# Total train data {len(train_split)}, val data {len(val_split)}')
    train_data = GraphDataset(train_split, edge_nfeats=edge_nfeats)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=train_batch_size, num_workers=0, pin_memory=False, shuffle=True)
    valid_data = GraphDataset(val_split, edge_nfeats=edge_nfeats)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=val_batch_size, num_workers=0, pin_memory=False, shuffle=False)

    if args.resume:
        print('Resuming training...')
        start_step = load_checkpoint(policy, optimizer, step='max', save_dir=load_model_dir, device=device)
        policy = policy.to(device)
    else:
        start_step = 0

    train(policy, optimizer, train_loader, device, nb_epochs, start_step=start_step, 
          eval_every=eval_every, save_every=save_every, 
          model_dir=model_dir, log_dir=log_dir, val_dataloader=valid_loader)