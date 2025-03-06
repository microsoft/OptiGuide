import argparse
import os
import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, QuantoConfig
import diskcache as dc
from contrast_dataset import getDataloadersWithCodeTexts

from gap_model import MyGNNAttn

model_name = 'nvidia/NV-Embed-v1'


def load_text_encoder(model_name: str):
    print(colored("TEXT ENCODER LOADING...", "red"))
    quantization_config = QuantoConfig(weights="int8")
    text_encoder = AutoModel.from_pretrained(model_name, 
                                             trust_remote_code=True,     
                                             quantization_config= quantization_config)
    text_encoder = text_encoder.cuda()
    return text_encoder

text_encoder = None

def encode_with_diskcache(model_name: str, texts: list):
    global text_encoder, args
    cache = dc.Cache(".llm_cache_dir")
    assert isinstance(texts, list)
    ans = []
    for text in texts:
        if (model_name, text) in cache:
            ans.append(cache[(model_name, text)])
        else:
            # only load the text encoder when necessary
            if text_encoder is None:
                text_encoder = load_text_encoder(model_name)
            with torch.no_grad():
                text_encoder = text_encoder.eval()
                outputs = text_encoder.encode([text], instruction="", max_length=512)
            outputs = outputs.detach().cpu().numpy().reshape(-1).tolist()
            cache[(model_name, text)] = outputs 
            ans.append(outputs)
    
    ans =  torch.tensor(ans)
    return ans.reshape(len(texts), -1)

def run(mode, epoch, text_encoder, milp_encoder, text_optimizer, milp_optimizer, dataloader,
        device='cuda', freeze_text_encoder=True, repeats=1, print_iters=10, writer=None):
    global OUT
    # Check if mode is valid
    if mode not in ["train", "validation"] and "test" not in mode:
        raise ValueError("Mode must be either 'train' or 'validation' or contains 'test'.")

    # Loss function (used in both training and validation modes)
    criterion = nn.CrossEntropyLoss()

    if not freeze_text_encoder:
        text_encoder.train() if mode == "train" else text_encoder.eval()
    milp_encoder.train() if mode == "train" else milp_encoder.eval()

    epoch_loss = 0.0
    running_n = 0
    i = 0
    accs = {"i2t": [], "t2i": [], "4way-i2t":[], "4way-t2i":[]}

    color = "green" if mode == "train" else "blue"
    
    with torch.set_grad_enabled(mode == "train"):
        for repeat in range(repeats):
            for batch in tqdm.tqdm(dataloader):
                text_inputs = batch.code_text
                images = batch
                running_n += len(text_inputs)
                i += 1

                # Move data to the appropriate device
                images = images.to(device)

                # Zero the parameter gradients (only in training mode)
                if mode == "train" and not freeze_text_encoder:
                    text_optimizer.zero_grad()
                    text_features = text_encoder.encode(text_inputs, instruction="", max_length=512) # [bs, 4096]
                else:
                    text_features = encode_with_diskcache(model_name, text_inputs) # [bs, 4096]
                text_features = text_features.to(device)
                text_features = text_features.float()
                
                if mode == "train":
                    milp_optimizer.zero_grad()

                assert text_features.shape == (len(text_inputs), 4096)

                # Forward pass through the encoders
                milp_features = milp_encoder(images) # [bs, n_out_neurons]

                # Normalize the features
                text_features = F.normalize(text_features, p=2, dim=1)
                milp_features = F.normalize(milp_features, p=2, dim=1)

                # Calculate the logits (dot product of text and image features)
                logits_per_image = milp_features @ text_features.T
                logits_per_text = text_features @ milp_features.T

                # Labels for contrastive learning
                labels = torch.arange(len(text_inputs)).to(device)

                # Calculate loss
                loss_i2t = criterion(logits_per_image, labels)
                loss_t2i = criterion(logits_per_text, labels)
                loss = (loss_i2t + loss_t2i) / 2

                # Calculate accuracy
                acc_i2t = (torch.argmax(logits_per_image, dim=1) == labels).float().mean()
                acc_t2i = (torch.argmax(logits_per_text, dim=1) == labels).float().mean()
                accs["i2t"].append(acc_i2t.item())
                accs["t2i"].append(acc_t2i.item())

                # calculate 4-way accuracy
                if len(text_inputs) >= 4:
                    milp_features_4way = milp_features[:4, :]
                    text_features_4way = text_features[:4, :]
                    logits_per_milp_4way = milp_features_4way @ text_features_4way.T
                    logits_per_text_4way = text_features_4way @ milp_features_4way.T
                    labels_4way = torch.arange(4).to(device)
                    acc_4way_i2t = (torch.argmax(logits_per_milp_4way, dim=1) == labels_4way).float().mean()
                    acc_4way_t2i = (torch.argmax(logits_per_text_4way, dim=1) == labels_4way).float().mean()
                    accs["4way-i2t"].append(acc_4way_i2t.item())
                    accs["4way-t2i"].append(acc_4way_t2i.item())

                if mode == "train":
                    # Backward pass and optimization
                    loss.backward()
                    if not freeze_text_encoder:
                        text_optimizer.step()
                    milp_optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()

                if i % print_iters == 0 and i > 0:
                    print(f'Epoch {epoch}, Loss: {epoch_loss/running_n}')
                    print(colored(f'Accuracy i2t: {sum(accs["i2t"])/len(accs["i2t"])}', color))
                    print(colored(f'Accuracy t2i: {sum(accs["t2i"])/len(accs["t2i"])}', color))
                    print(colored(f'4-way Accuracy i2t: {sum(accs["4way-i2t"])/len(accs["4way-i2t"])}', color))
                    print(colored(f'4-way Accuracy t2i: {sum(accs["4way-t2i"])/len(accs["4way-t2i"])}', color))

        print(args)
        print(colored(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader)}', color))
        print(colored(f'Accuracy i2t: {sum(accs["i2t"])/len(accs["i2t"])}', color))
        print(colored(f'Accuracy t2i: {sum(accs["t2i"])/len(accs["t2i"])}', color))
        print(colored(f'4-way Accuracy i2t: {sum(accs["4way-i2t"])/len(accs["4way-i2t"])}', color))
        print(colored(f'4-way Accuracy t2i: {sum(accs["4way-t2i"])/len(accs["4way-t2i"])}', color))

        OUT.write(f'{mode} Epoch {epoch}, Loss: {epoch_loss/len(dataloader)}\n')
        OUT.write(f'{mode} Epoch {epoch}, Accuracy i2t: {sum(accs["i2t"])/len(accs["i2t"])}\n')
        OUT.write(f'{mode} Epoch {epoch},Accuracy t2i: {sum(accs["t2i"])/len(accs["t2i"])}\n')
        OUT.write(f'{mode} Epoch {epoch},4-way Accuracy i2t: {sum(accs["4way-i2t"])/len(accs["4way-i2t"])}\n')
        OUT.write(f'{mode} Epoch {epoch},4-way Accuracy t2i: {sum(accs["4way-t2i"])/len(accs["4way-t2i"])}\n\n\n')
        OUT.flush()

        # Add final TensorBoard logging for the epoch
        if writer:
            writer.add_scalar(f'{mode}/Loss', epoch_loss/len(dataloader), epoch)
            writer.add_scalar(f'{mode}/Acc_i2t', sum(accs["i2t"])/len(accs["i2t"]), epoch)
            writer.add_scalar(f'{mode}/Acc_t2i', sum(accs["t2i"])/len(accs["t2i"]), epoch)
            writer.add_scalar(f'{mode}/4way_Acc_i2t', sum(accs["4way-i2t"])/len(accs["4way-i2t"]), epoch)
            writer.add_scalar(f'{mode}/4way_Acc_t2i', sum(accs["4way-t2i"])/len(accs["4way-t2i"]), epoch)


#### Add argparser  
parser = argparse.ArgumentParser(description="Train the GNN model.")
parser.add_argument("--model_name", type=str, default="nvidia/NV-Embed-v1", help="The name of the model to use.")
parser.add_argument("--epochs", type=int, default=5, help="The number of epochs to train for.")
parser.add_argument("--embed_size", type=int, default=128, help="The embedding size for the model.")
parser.add_argument("--num_milp_instance", type=int, default=20000, help="The number of MILP instances to use.")
parser.add_argument("--num_milp_class", type=int, default=20000, help="The number of MILP instances to use.")
parser.add_argument("--max_token_attn", type=int, default=128, help="The maximum number of tokens for attention.")
parser.add_argument("--n_attn_layers", type=int, default=1, help="The number of attention layers.")
parser.add_argument("--n_gnn_layers", type=int, default=1, help="The number of GNN layers.")
parser.add_argument("--load_from", type=str, default=None, help="Path to load the model from.")
parser.add_argument("--save_to", type=str, default=None, help="Path to save the model to.")
parser.add_argument("--dataset", type=str, default="ours", help="The dataset to use for training.")
parser.add_argument("--eval_epochs", type=int, default=1, help="The number of epochs between evaluations.")
parser.add_argument("--print_iters", type=int, default=10, help="The number of iterations between printing results.")
parser.add_argument("--lr", type=float, default=0.00005, help="The learning rate for the optimizer.")
parser.add_argument("--text_types", type=str, default="all", help="Either `all` or `description only`")
parser.add_argument("--log_root", type=str, default="save_data/contrast", help="Root for loggin and saving.")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for the network.`")


args = parser.parse_args()
model_name = args.model_name

assert args.text_types in ["all", "description only"] # description only means we do not include code.

if args.dataset == "ours":
    train_dataloader = getDataloadersWithCodeTexts("train_ours_data.pkl.gz",  
                                        num_milp_instance=args.num_milp_instance,
                                        num_milp_class=args.num_milp_class,
                                        batch_size=10, shuffle_flag=True, 
                                        pin_memory=False, text_types=args.text_types)
    valid_dataloader = None
    validation_repeats = 1
elif args.dataset == "ours+seed":
    train_dataloader = getDataloadersWithCodeTexts(["train_ours_data.pkl.gz", "train_seed_data.pkl.gz"],  
                                        num_milp_instance=args.num_milp,
                                        num_milp_class=args.num_milp_class,
                                        batch_size=10, shuffle_flag=True, 
                                        pin_memory=False, text_types=args.text_types)
    valid_dataloader = None
    validation_repeats = 1
elif args.dataset == "miplib":
    print("args.num_milp is ignored because the dataset is MIPLIB")
    print("We split MIPLIB to train and test here for validation.")
    train_dataloader = getDataloadersWithCodeTexts("train_miplib_data.pkl.gz",
                                        batch_size=4, shuffle_flag=True,  # TODO: change back to 10.
                                        pin_memory=False, )
    valid_dataloader = getDataloadersWithCodeTexts("test_miplib_data.pkl.gz", 
                                        batch_size=10, shuffle_flag=True, 
                                        pin_memory=False, )
    validation_repeats = 10 # repeat the validation multiple times to achieve a stable number
elif "seed" in args.dataset:
    print("args.num_milp is ignored because the dataset is SEED")
    print("We use both the train and test data for training! Because it is Seed baseline.")
    train_dataloader = getDataloadersWithCodeTexts( [f"train_{args.dataset}_data.pkl.gz",
            f"test_{args.dataset}_data.pkl.gz",],
                                        batch_size=10, shuffle_flag=True, 
                                        pin_memory=False, )
    valid_dataloader = None
    validation_repeats = 1
else:
    pdb.set_trace()
    raise ValueError("Invalid dataset.")

lr = args.lr


if args.dataset != "miplib":
    test_ours_dataloader = getDataloadersWithCodeTexts("test_data.pkl.gz", 
                                        batch_size=10, shuffle_flag=True, 
                                        pin_memory=False, text_types=args.text_types)
else:
    test_ours_dataloader = None

if args.dataset != "miplib":
    # we only test out-domain performance if the training data is not miplib.
    test_miplib_loader = getDataloadersWithCodeTexts("miplib_data.pkl.gz", 
                                            batch_size=10, shuffle_flag=True, 
                                            pin_memory=False,)
else:
    test_miplib_loader = None

# MAIN #
milp_encoder =  MyGNNAttn(emb_size=args.embed_size, n_out_neurons=4096, dropout=args.dropout, max_token_attn=args.max_token_attn,  
                    n_attn_iters=args.n_attn_layers, n_gnn_iters=args.n_gnn_layers, edge_nfeats=1)

if args.load_from and args.load_from != "None.pt":
    state_dict = torch.load(os.path.join(args.log_root, args.load_from))
    milp_encoder.load_state_dict(state_dict)


milp_encoder =  milp_encoder.cuda()
# print the number of trainable parameters, split with comma in thousands
n_trainable_params = sum(p.numel() for p in milp_encoder.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {n_trainable_params:,}")

# Optimizers for the encoders (only used in training mode)
text_optimizer = None # optim.Adam(text_encoder.parameters(), lr=0.0001)
milp_optimizer = optim.Adam(milp_encoder.parameters(), lr=lr)


OUT_FILE = os.path.join(args.log_root, f"{args.dataset}_use_attn_embed_{args.embed_size}_num_milp_{args.num_milp_instance}-{args.num_milp_class}_layser_{args.n_attn_layers}_{args.n_gnn_layers}_output.txt")
OUT = open(OUT_FILE, "a")
OUT.write(time.ctime())
OUT.write("args: " + str(args) + "\n")
OUT.write("n_trainable_params: " + str(n_trainable_params) + "\n")
OUT.flush()

# Create a SummaryWriter instance
if args.load_from:
    llf = str(os.path.basename(args.load_from))
else:
    llf = "None"
writer = SummaryWriter(log_dir=os.path.join(args.log_root, "tensorboard_logs_emb64",  
                        f"{args.dataset}",
                        f"from{llf}_use_attn_" + 
                        f"embed_{args.embed_size}_num_milp_{args.num_milp_instance}-{args.num_milp_class}_{time.time():2f}"))
writer.add_text("args", str(args))

if args.dataset == "miplib" and valid_dataloader:
    # let's do zero-shot eval first.
    run("validation", -1, text_encoder, milp_encoder, text_optimizer, milp_optimizer, valid_dataloader,
            device='cuda', freeze_text_encoder=True, repeats=validation_repeats, print_iters=args.print_iters, writer=writer) 

for epoch in range(args.epochs):
    run("train", epoch, text_encoder, milp_encoder, text_optimizer, milp_optimizer, train_dataloader,
             device='cuda', freeze_text_encoder=True, print_iters=args.print_iters, writer=writer) 

    torch.save(milp_encoder.state_dict(), 
            os.path.join(args.log_root, 
            f"{args.dataset}_milp_encoder_use_attn_embed_{args.embed_size}_num_milp_{args.num_milp_instance}-{args.num_milp_class}_epoch{epoch}.pt"))

    if args.save_to:
        torch.save(milp_encoder.state_dict(), os.path.join(args.log_root, args.save_to))

    if epoch == args.epochs - 1 or epoch % args.eval_epochs == 0:
        if valid_dataloader:
            run("validation", epoch, text_encoder, milp_encoder, text_optimizer, milp_optimizer, valid_dataloader,
                device='cuda', freeze_text_encoder=True, repeats=validation_repeats, print_iters=args.print_iters, writer=writer) 

        if test_ours_dataloader:
            run("test_ours", epoch, text_encoder, milp_encoder, text_optimizer, milp_optimizer, test_ours_dataloader,
                device='cuda', freeze_text_encoder=True, repeats=1, print_iters=args.print_iters, writer=writer) 

        if test_miplib_loader:
            run("test_miplib", epoch, text_encoder, milp_encoder, text_optimizer, milp_optimizer, test_miplib_loader,
                device='cuda', freeze_text_encoder=True, repeats=1, print_iters=args.print_iters, writer=writer) 

# Close the SummaryWriter
writer.close()

