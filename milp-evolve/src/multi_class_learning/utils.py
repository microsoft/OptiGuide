import json
import numpy as np
import os
import re
import random
import pickle
import joblib
import torch

import tqdm
import glob
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from joblib import Parallel, delayed
import gzip
from scipy.stats import gmean as scipy_gmean

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
milp_evolve_llm_dir = os.path.abspath(os.path.join(current_dir, '..', 'milp_evolve_llm'))
sys.path.append(milp_evolve_llm_dir)

from src.utils import (
    remove_duplicate, 
    get_status_time_and_gap, 
    add_time_limit, 
    run_script_from_tmp, 
    check_valid_file,
    check_feasibility, 
    change_seed_in_file, 
    change_seed_and_check_feasibility, 
    change_seed_and_check_valid_file
)


def gmean(arr, zero=1e-5, shift=1):
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    return scipy_gmean(np.maximum(arr + shift, zero)) - shift


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def multiprocess(func, tasks, n_cpus=None):
    if n_cpus == 1 or len(tasks) == 1:
        return [func(t) for t in tasks]
    with Pool(n_cpus or os.cpu_count()) as pool:
        return list(pool.imap(func, tasks))

def multithread(func, tasks, n_cpus=None, show_bar=True):
    bar = lambda x: tqdm(x, total=len(tasks)) if show_bar else x
    if n_cpus == 1 or len(tasks) == 1:
        return [func(t) for t in bar(tasks)]
    with ThreadPool(n_cpus or os.cpu_count()) as pool:
        return list(bar(pool.imap(func, tasks)))

def parallel_fn(func, tasks, n_cpus=None):
    if len(tasks) == 0:
        return []
    if n_cpus == 1 or len(tasks) == 1:
        return [func(t) for t in tasks]
    n_cpus = os.cpu_count() if n_cpus is None else min(n_cpus, os.cpu_count()) 
    return Parallel(n_jobs=min(n_cpus, len(tasks)))(delayed(func)(t) for t in tasks)

def save_state(path, state):
    os.makedirs(path, exist_ok=True)

    for k, obj in state.items():
        obj_path = os.path.join(path, k)

        if isinstance(obj, np.ndarray):
            save_func = save_numpy
            obj_path += '.npy'
        else:
            save_func = save_pickle
            obj_path += '.pkl'

        save_func(obj_path, obj)


def save_json(path, d):
    dir_path, file_name = os.path.split(path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(path, 'w') as file:
        json.dump(d, file, indent=4)

def load_json(path, default=[]):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        d = json.load(f)
    return d

def save_torch(file_path, data):
    dir_path, file_name = os.path.split(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    torch.save(data, file_path)

def load_torch(file_path, map_location=None):
    if map_location:
        data = torch.load(file_path, map_location=map_location)    
    else:
        data = torch.load(file_path)
    
    return data

def save_numpy(path, arr):
    dir_path, file_name = os.path.split(path)
    os.makedirs(dir_path, exist_ok=True)
    with open(path, 'wb') as file:
        np.save(file, arr)

def load_numpy(path):
    with open(path, 'rb') as file:
        arr = np.load(path)
    return arr

def get_compress_path(path):
    return path + '.gz' if not path.endswith('.gz') else path


def save_gzip(path, obj):
    gzip_path = get_compress_path(path)
    dir_path, file_name = os.path.split(gzip_path)
    os.makedirs(dir_path, exist_ok=True)

    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump(obj, file)  # , protocol=5


def load_gzip(path):
    gzip_path = get_compress_path(path)
    if os.path.exists(path) and not os.path.exists(gzip_path):
        return load_pickle(path)
    
    with gzip.open(gzip_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def save_pickle(path, obj):
    dir_path, file_name = os.path.split(path)
    os.makedirs(dir_path, exist_ok=True)

    with open(path, 'wb') as file:
        pickle.dump(obj, file)   

def load_pickle(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj

def save_joblib(path, obj):
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)


class ExitStatus:
    CONVERGED_DUAL = 0
    CONVERGED_INTEGRAL = 1
    MAXITER = 2
    ERROR_TRIVIAL = 3
    ERROR_SEPARATION = 4
    ERROR_NOCUTS = 5
    ERROR_DUAL = 6
    ERROR_SIGN_SWITCH = 7
    ERROR_IGC_MESS = 8
    ERROR_DEF_JSON_MISSING = 9
    ERROR_EXP_JSON_MISSING = 10


"""
Utility functions to load and save torch model checkpoints 
"""
def load_checkpoint(model, optimizer=None, step='max', save_dir='checkpoints', device='cpu', exclude_keys=[]):
    os.makedirs(save_dir, exist_ok=True)

    checkpoints = [x for x in os.listdir(save_dir) if not x.startswith('events') and not x.endswith('.json') and not x.endswith('.pkl')]

    if step == 'max':
        step = 0
        if checkpoints:
            step, last_checkpoint = max([(int(x.split('.')[0]), x) for x in checkpoints])
    else:
        last_checkpoint = str(step) + '.pth'
    
    if step:
        save_path = os.path.join(save_dir, last_checkpoint)
        state = torch.load(save_path, map_location=device)

        if len(exclude_keys) > 0:
            model_state = state['model'] if 'model' in state else state
            model_state = {k: v for k, v in model_state.items() if not any(k.startswith(exclude_key) for exclude_key in exclude_keys)}
            model.load_state_dict(model_state, strict=False)
            
            if optimizer and 'optimizer' in state:
                optimizer_state_dict = state['optimizer']
                excluded_param_ids = {
                    id(param) for name, param in model.named_parameters() if any(name.startswith(exclude_key) for exclude_key in exclude_keys)
                }
                optimizer_state_dict['state'] = {k: v for k, v in optimizer_state_dict['state'].items() if k not in excluded_param_ids}
                optimizer.load_state_dict(optimizer_state_dict)
        else:
            model_state = state['model'] if 'model' in state else state
            model.load_state_dict(model_state)
            if optimizer and 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
        
        print('Loaded checkpoint %s' % save_path)
    
    return step

def save_checkpoint(model, step, optimizer=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(step) + '.pth')

    if optimizer is None:
        torch.save(dict(model=model.state_dict()), save_path)
    else:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), save_path)
    print('Saved checkpoint %s' % save_path)


def get_train_val_test_splits(files, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1
    N_problems = len(files)
    train_size = int(train_ratio * N_problems)
    val_size = int(val_ratio * N_problems)
    random.shuffle(files)
    train_split = files[:train_size]
    val_split = files[train_size:train_size + val_size]
    test_split = files[train_size + val_size:]
    return train_split, val_split, test_split



def print_dash_str(s='', dash_str_len=120):
    if len(s) == 0:
        print(''.join(['-' for _ in range(dash_str_len)]))
        return 
    if len(s) > dash_str_len:
        print(s)
        return
    
    n_left_dash = (dash_str_len - len(s) - 2) // 2
    n_right_dash = dash_str_len - len(s) - 2 - n_left_dash
    print(''.join(['-' for _ in range(n_left_dash)]) + ' ' + s + ' ' + ''.join(['-' for _ in range(n_right_dash)]))



#### get paths functions
def get_basename(difficulty, code_start_idx, code_end_idx, code_exclude_idxs,
                 code_idx_difficulty_list, code_str='code'):
    
        
    def get_difficulty_str(difficulty):
        if 'easy' in difficulty:
            return 'E'
        elif 'medium' in difficulty:
            return 'M'
        elif 'hard' in difficulty:
            return 'H'
        elif difficulty == 'all':
            return 'all'
        raise ValueError(f'Unknown difficulty {difficulty}')

    def get_code_exclude_str(code_exclude_idxs):
        return '' if len(code_exclude_idxs) == 0 else f'_exclude-' + "-".join([str(idx) for idx in code_exclude_idxs])

    if isinstance(difficulty, str): difficulty = [difficulty]
    if difficulty == ['easy', 'medium', 'hard']: difficulty = ['all']

    basenames = []
    code_exclude_idxs = sorted([idx for idx in code_exclude_idxs if idx in range(code_start_idx, code_end_idx)])
    if len(code_idx_difficulty_list) > 0:
        basename = f"{code_str}_" + "_".join([f'{get_difficulty_str(diff)}{code_start_idx}-{code_end_idx}' 
                                        for diff in code_idx_difficulty_list]) + get_code_exclude_str(code_exclude_idxs)
    else:
        if len(difficulty) == 1:
            basename = f'{code_str}_{difficulty[0]}{code_start_idx}-{code_end_idx}{get_code_exclude_str(code_exclude_idxs)}'
        else:
            basename = f"{code_str}_" + "_".join([f'{get_difficulty_str(diff)}_{code_start_idx}_{code_end_idx}' 
                                            for diff in difficulty]) + get_code_exclude_str(code_exclude_idxs)
    basenames.append(basename)

    basename = "_".join(basenames)

    return basename


def get_model_name_from_base_name(basename, args, parser, mode='train'):
    for arg_full, arg_short in [('emb_size', 'emb'), ('edge_nfeats', 'edge')]:
        if hasattr(args, arg_full):
            basename += f'_{arg_short}{getattr(args, arg_full)}'
    
    if hasattr(args, 'n_layers') and args.n_layers != 1:
        basename += f'_nlayer{args.n_layers}'

    if mode == 'train':
        ntrain_list = [('ntrain_instances', 'Ntrain'), ('ntrain_data', 'NtrainData'), ('max_data_per_instance', 'MaxData')]
    else:
        ntrain_list = [('model_ntrain_instances', 'Ntrain'), ('model_ntrain_data', 'NtrainData'), ('model_max_data_per_instance', 'MaxData')]

    for arg_full, arg_short in ntrain_list:
        if hasattr(args, arg_full):        
            arg_value = getattr(args, arg_full)
            default_value = parser.get_default(arg_full)
            if arg_value != default_value:
                basename += f'_{arg_short}{arg_value}'

    return basename

def get_model_name(args, parser, mode='train'):
    if mode == 'train':
        difficulty = args.difficulty
        code_start_idx, code_end_idx, code_exclude_idxs = args.code_start_idx, args.code_end_idx, args.code_exclude_idxs
        code_idx_difficulty_list = args.code_idx_difficulty_list
        code_str = getattr(args, 'code_str', 'code')
    else:
        difficulty = args.model_difficulty
        code_start_idx, code_end_idx, code_exclude_idxs = args.model_code_start_idx, args.model_code_end_idx, args.model_code_exclude_idxs
        code_idx_difficulty_list = args.model_code_idx_difficulty_list
        code_str = getattr(args, 'model_code_str', 'code')

    basename = get_basename(difficulty, code_start_idx, code_end_idx, code_exclude_idxs,
                                        code_idx_difficulty_list, code_str=code_str)
    
    modelname = get_model_name_from_base_name(basename, args, parser, mode)
    return modelname


def get_instances_dir_list(args):
    difficulty = args.difficulty
    # if difficulty is string
    if isinstance(difficulty, str): difficulty = [difficulty]
    code_str = getattr(args, 'code_str', 'code')

    instances_dir_list = []
    
    if len(args.code_idx_difficulty_list) > 0:
        instances_dir_list = [f'{code_str}/milp_{idx}-{diff}' for idx in range(args.code_start_idx, args.code_end_idx) for diff in args.code_idx_difficulty_list if idx not in args.code_exclude_idxs]
    else:
        if len(difficulty) == 1 and difficulty[0] == 'all':
            difficulty = ['easy', 'medium', 'hard']

        instances_dir_list += [f'{code_str}/milp_{idx}-{diff}' for idx in range(args.code_start_idx, args.code_end_idx) for diff in difficulty if idx not in args.code_exclude_idxs]

    # remove all instances_dir without any instances
    instances_dir_list = [instances_dir for instances_dir in instances_dir_list if len(get_files_by_extension(os.path.join(args.parent_instances_dir, instances_dir))) > 0]

    # if the instance filter dir exists: filter out all instances in the list
    if hasattr(args, 'filter_dir') and args.filter_dir:
        instance_dir_filter_json = os.path.join(args.parent_instances_metadata_dir, 'exclude_dir', f'exclude_instance_dir.json')
        if os.path.exists(instance_dir_filter_json):
            exclude_instance_dir = set(load_json(instance_dir_filter_json))
            instances_dir_list = [instances_dir for instances_dir in instances_dir_list if instances_dir not in exclude_instance_dir]
    return instances_dir_list


def get_collect_paths(args):
    instances_dir_list = get_instances_dir_list(args)
    instances_split_files = [os.path.join(args.parent_instances_metadata_dir, instances_dir, f'instances_split.json') for instances_dir in instances_dir_list]  # data_split.json for code
    generated_json_paths = [os.path.join(args.parent_data_metadata_dir, instances_dir, f'generated.json') for instances_dir in instances_dir_list]
    return instances_dir_list, generated_json_paths, instances_split_files


def get_train_paths(args, parser):
    instances_dir_list = get_instances_dir_list(args)
    modelname = get_model_name(args, parser, mode='train')
    log_dir = os.path.join(args.parent_log_dir, modelname)
    model_dir = os.path.join(args.parent_model_dir, modelname)
    instances_split_files = [os.path.join(args.parent_instances_metadata_dir, instances_dir, f'instances_split.json') for instances_dir in instances_dir_list]  
    return instances_dir_list, instances_split_files, model_dir, log_dir


def get_train_paths_by_dir(args, parser):
    code_str = getattr(args, 'code_str', 'code')
    basename = get_basename(args.difficulty, args.code_start_idx, args.code_end_idx, args.code_exclude_idxs,
                            args.code_idx_difficulty_list, code_str=code_str)
    instances_dir_split_file = os.path.join(args.parent_instances_metadata_dir, f'by_dir/{basename}/instances_split.json')
    modelname = get_model_name(args, parser, mode='train')
    log_dir = os.path.join(args.parent_log_dir, modelname)
    model_dir = os.path.join(args.parent_model_dir, modelname)
    instances_dir_list = get_instances_dir_list(args)

    return instances_dir_list, instances_dir_split_file, model_dir, log_dir


def get_rollout_paths(args, parser):
    instances_dir_list = get_instances_dir_list(args)
    instances_split_files = [os.path.join(args.parent_instances_metadata_dir, instances_dir, f'instances_split.json') for instances_dir in instances_dir_list] 
    modelname = get_model_name(args, parser, mode='test')
    model_dir = os.path.join(args.parent_model_dir, modelname)
    parent_test_stats_dir = os.path.join(args.parent_test_stats_dir, modelname)

    return instances_dir_list, instances_split_files, model_dir, parent_test_stats_dir


def get_rollout_paths_by_dir(args, parser):   
    code_str = getattr(args, 'code_str', 'code') 
    basename = get_basename(args.difficulty, args.code_start_idx, args.code_end_idx, args.code_exclude_idxs,
                            args.code_idx_difficulty_list, code_str=code_str)
    instances_dir_list = get_instances_dir_list(args)
    instances_dir_split_file = os.path.join(args.parent_instances_metadata_dir, f'by_dir/{basename}/instances_split.json')
    modelname = get_model_name(args, parser, mode='test')
    model_dir = os.path.join(args.parent_model_dir, modelname)
    parent_test_stats_dir = os.path.join(args.parent_test_stats_dir, modelname, 'by_dir')
    return instances_dir_list, instances_dir_split_file, model_dir, parent_test_stats_dir

def get_test_paths(args, parser):
    instances_dir_list = get_instances_dir_list(args)
    instances_split_files = [os.path.join(args.parent_instances_metadata_dir, instances_dir, f'instances_split.json')  for instances_dir in instances_dir_list] 
    modelname = get_model_name(args, parser, mode='test')
    model_dir = os.path.join(args.parent_model_dir, modelname)
    parent_test_stats_dir = os.path.join(args.parent_test_stats_dir, modelname)

    return instances_dir_list, instances_split_files, model_dir, parent_test_stats_dir


def get_test_paths_by_dir(args, parser):
    code_str = getattr(args, 'code_str', 'code')
    basename = get_basename(args.difficulty, args.code_start_idx, args.code_end_idx, args.code_exclude_idxs,
                            args.code_idx_difficulty_list, code_str=code_str)
    instances_dir_split_file = os.path.join(args.parent_instances_metadata_dir, f'by_dir/{basename}/instances_split.json')
    modelname = get_model_name(args, parser, mode='test')
    model_dir = os.path.join(args.parent_model_dir, modelname)
    instances_dir_list = get_instances_dir_list(args)
    parent_test_stats_dir = os.path.join(args.parent_test_stats_dir, modelname)

    return instances_dir_list, instances_dir_split_file, model_dir, parent_test_stats_dir


# Split instances
def instance_numerical_sort_key(value):
    filename = os.path.basename(value)
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part for part in parts]

def branching_data_numerical_sort_key(value):
    filename = os.path.basename(value)
    parts = filename.split('_')
    return int(parts[-1].split('.')[0])


def replace_extension(basename):
    # Determine the extension and remove it
    extensions = ['.mps', '.mps.gz', '.lp', '.proto.lp', '.lp.gz', '.proto.lp.gz']
    for ext in extensions:
        if basename.endswith(ext):
            instance_idx = basename.replace(ext, '')
            return instance_idx
    raise ValueError("Unknown file extension")


def get_files_by_extension(file_dir, file_prefix='*'):
    extensions = ['.mps', '.mps.gz', '.lp', '.proto.lp', '.lp.gz', '.proto.lp.gz']

    files = []
    
    for ext in extensions:
        files.extend(glob.glob(os.path.join(file_dir, file_prefix + ext)))
    
    return files

def get_gzip_files_by_extension(file_dir, file_prefix='*', ext='pkl'):
    extensions = [f'.{ext}', f'.{ext}.gz']

    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(file_dir, file_prefix + ext)))
    return files


### load instance split, assume split file exists
def load_instances_split(instances_split_file, split='train', N_instances=float('inf'), select_option='first'):
    # assert os.path.exists(instances_split_file), 'Instance split file does not exist'
    split_dict = load_json(instances_split_file)
    if not split_dict:
        return []
    instances_split = sorted(split_dict[split], key=instance_numerical_sort_key)

    if len(instances_split) == 0:
        print('No instances split found. The user may need to generate a new split.')
        return []
        
    # filter by # instances
    if N_instances < len(instances_split):
        if select_option == 'first':
            instances_split = instances_split[:N_instances]
        else:
            instances_split = random.sample(instances_split, N_instances)
    return instances_split


### get instance split, assume split file may not exist or it may not contain all instances
def get_instances_train_val_test_split(instances_split_file, parent_instances_dir, instances_dir, select_option='first',
                                      Ntrain_instances=float('inf'), Nval_instances=float('inf'), Ntest_instances=float('inf'),
                                      regenerate_split=False):
    train_split, val_split, test_split = [], [], []
    instance_files = get_files_by_extension(os.path.join(parent_instances_dir, instances_dir))
    
    # instance_files = [instance_file.replace(parent_instances_dir, '').lstrip('/') for instance_file in instance_files]
    if len(instance_files) == 0:
        return [], [], []
    if not os.path.exists(instances_split_file):
        # generated instance split
        train_split, val_split, test_split = get_train_val_test_splits(instance_files)
        # sort the files
        train_split = sorted(train_split, key=instance_numerical_sort_key)
        val_split = sorted(val_split, key=instance_numerical_sort_key)
        test_split = sorted(test_split, key=instance_numerical_sort_key)
        save_json(instances_split_file, {'train': train_split, 'val': val_split, 'test': test_split})
    else:
        # update train, val, test split to check if we miss instanecs
        split_dict = load_json(instances_split_file)
        if not split_dict:
            return [], [], []
        train_split, val_split, test_split = split_dict['train'], split_dict['val'], split_dict['test']
        
        if regenerate_split:
            # filter out nonexisting files
            # train_split: get absolute path of train_split path (no concate, directly get absolute path)
            train_split = [os.path.relpath(instance_path) for instance_path in train_split if os.path.exists(instance_path)]
            val_split = [os.path.relpath(instance_path) for instance_path in val_split if os.path.exists(instance_path)]
            test_split = [os.path.relpath(instance_path) for instance_path in test_split if os.path.exists(instance_path)]
            instance_files = [os.path.relpath(instance_path) for instance_path in instance_files if os.path.exists(instance_path)]
            train_split_set, val_split_set, test_split_set = set(train_split), set(val_split), set(test_split)
            paths_to_problems_not_split = list(set(instance_files) - set(train_split) - set(val_split) - set(test_split))
            
            if len(paths_to_problems_not_split) > 0 or len(train_split_set) < len(train_split) or len(val_split_set) < len(val_split) or len(test_split_set) < len(test_split):
                train_split_, val_split_, test_split_ = get_train_val_test_splits(paths_to_problems_not_split)
                train_split, val_split, test_split = list(set(train_split + train_split_)), list(set(val_split + val_split_)), list(set(test_split + test_split_))
                # sort the files
                train_split = sorted(train_split, key=instance_numerical_sort_key)
                val_split = sorted(val_split, key=instance_numerical_sort_key)
                test_split = sorted(test_split, key=instance_numerical_sort_key)
                save_json(instances_split_file, {'train': train_split, 'val': val_split, 'test': test_split})
    # if filter_instance: exclude certain instances
    if Ntrain_instances < len(train_split):
        if select_option == 'first':
            train_split = train_split[:Ntrain_instances]
        else:
            train_split = random.sample(train_split, Ntrain_instances)
    if Nval_instances < len(val_split):
        if select_option == 'first':
            val_split = val_split[:Nval_instances]
        else:
            val_split = random.sample(val_split, Nval_instances)
    if Ntest_instances < len(test_split):
        if select_option == 'first':
            test_split = test_split[:Ntest_instances]
        else:
            test_split = random.sample(test_split, Ntest_instances)
    return train_split, val_split, test_split


### get instance directory split, save the split
def get_instances_dir_train_val_test_split(instances_dir_split_file, instances_dir_list, regenerate_split=False):
    train_instances_dir_split, val_instances_dir_split, test_instances_dir_split = [], [], []
    if os.path.exists(instances_dir_split_file):
        instances_dir_split = load_json(instances_dir_split_file)
        if instances_dir_split:
            train_instances_dir_split, val_instances_dir_split, test_instances_dir_split = instances_dir_split['train'], instances_dir_split['val'], instances_dir_split['test']
       
    if regenerate_split:
        instances_dir_not_split = list(set(instances_dir_list) - set(train_instances_dir_split) - set(val_instances_dir_split) - set(test_instances_dir_split))

        if len(instances_dir_not_split) > 0:
            train_instances_dir_split_, val_instances_dir_split_, test_instances_dir_split_ = get_train_val_test_splits(instances_dir_not_split)
            train_instances_dir_split = sorted(train_instances_dir_split + train_instances_dir_split_, key=instance_numerical_sort_key)
            val_instances_dir_split = sorted(val_instances_dir_split + val_instances_dir_split_, key=instance_numerical_sort_key)
            test_instances_dir_split = sorted(test_instances_dir_split + test_instances_dir_split_, key=instance_numerical_sort_key)
            save_json(instances_dir_split_file, {'train': train_instances_dir_split, 'val': val_instances_dir_split, 'test': test_instances_dir_split})

    return train_instances_dir_split, val_instances_dir_split, test_instances_dir_split


def load_instances_by_dir(parent_instances_dir, instances_dir, N_instances=float('inf'), select_option='first'):
    # load all instances in the directory
    instances_files = sorted(get_files_by_extension(os.path.join(parent_instances_dir, instances_dir)), key=instance_numerical_sort_key)
    if len(instances_files) == 0:
        print('No instances split found. The user may need to generate a new split.')
        return []
    
    if N_instances < len(instances_files):
        if select_option == 'first':
            instances_files = instances_files[:N_instances]
        else:
            instances_files = random.sample(instances_files, N_instances)
    return instances_files


def get_instances_list_by_dir(parent_instances_dir, instances_dir_split, select_option='first', N_instances=float('inf')):
    data_split = []
    for instances_dir in instances_dir_split:  #  tqdm.tqdm(instances_dir_split):         
        instances_dir = instances_dir.replace(parent_instances_dir, '').lstrip('/')   
        instances_files = load_instances_by_dir(parent_instances_dir, instances_dir, N_instances, select_option)
        print(f'Load {len(instances_files)} from {os.path.join(parent_instances_dir, instances_dir)} ...')
        data_split.extend(instances_files)

    return data_split


### get instance split based on the directories
def get_instances_train_val_test_split_list_by_dir(instances_dir_split_file, parent_instances_dir, parent_instances_metadata_dir, instances_dir_list, 
                                                   train_instances_dir_split=[], val_instances_dir_split=[], test_instances_dir_split=[], 
                                                   select_option='first', Ntrain_instances=float('inf'), Nval_instances=float('inf'), Ntest_instances=float('inf')):
    # check if instances_dir in train_instances_dir_split, val_instances_dir_split, test_instances_dir_split all exists
    joint_split = train_instances_dir_split + val_instances_dir_split + test_instances_dir_split
    for instances_dir in joint_split:
        if not os.path.exists(instances_dir):
            print(f'{instances_dir} does not exist. The user may need to re-split the instances ...')
    
    if len(joint_split) == 0:
        train_instances_dir_split, val_instances_dir_split, test_instances_dir_split = get_instances_dir_train_val_test_split(instances_dir_split_file, instances_dir_list)

    train_split, val_split, test_split = [], [], []
    for split, instances_dir_split in [('train', train_instances_dir_split), ('val', val_instances_dir_split), ('test', test_instances_dir_split)]:
        print(f'{split}: {len(instances_dir_split)} to check ...')

        for instances_dir in instances_dir_split:  #  tqdm.tqdm(instances_dir_split):   
            instances_dir = instances_dir.replace(parent_instances_dir, '').lstrip('/')          
            instances_split_file = os.path.join(parent_instances_metadata_dir, instances_dir, f'instances_split.json')
            if not os.path.exists(instances_split_file):
                get_instances_train_val_test_split(instances_split_file, parent_instances_dir, instances_dir, select_option=select_option,
                                                    Ntrain_instances=Ntrain_instances, Nval_instances=Nval_instances, Ntest_instances=Ntest_instances)
            
            train_split_ = load_instances_split(instances_split_file, 'train', Ntrain_instances, select_option)
            val_split_ = load_instances_split(instances_split_file, 'val', Nval_instances, select_option)
            test_split_ = load_instances_split(instances_split_file, 'test', Ntest_instances, select_option)

            if split == 'train':
                train_split.extend(train_split_ + val_split_ + test_split_)
            elif split == 'val':
                val_split.extend(train_split_ + val_split_ + test_split_)
            else:
                test_split.extend(train_split_ + val_split_ + test_split_)

    return train_split, val_split, test_split


def get_data_by_instance_split_branching(instances_split_file, parent_instances_dir, parent_data_dir, split='train', 
                                         select_option='first', N_instances=float('inf'), max_data_per_instance=float('inf'), 
                                         N_data=float('inf')):

    if N_instances == 0 or N_data == 0 or not os.path.exists(instances_split_file):
        return []
    
    assert (N_instances == float('inf') or N_data == float('inf')), 'Should only set at most one of N_instances or N_data'
    
    instances_split = load_instances_split(instances_split_file, split, float('inf'), select_option)

    if len(instances_split) == 0:
        return []

    paths_to_datafiles = []
    num_loaded = 0
    for path in instances_split: 
        basename = os.path.basename(path)
        instance_dir = os.path.dirname(path).replace(parent_instances_dir, '').replace(basename, '').lstrip('/') 
        instance_idx = replace_extension(basename)

        data_files = sorted(get_gzip_files_by_extension(os.path.join(parent_data_dir, instance_dir), 
                                                          file_prefix=f'data_{instance_idx}_*'), key=branching_data_numerical_sort_key)
        data_files = data_files[:max_data_per_instance] if max_data_per_instance < len(data_files) else data_files
        paths_to_datafiles.extend(data_files)
        if len(data_files) > 0:
            num_loaded += 1
        if num_loaded >= N_instances or (select_option == 'first' and len(paths_to_datafiles) >= N_data):
            break

    if N_data < len(paths_to_datafiles):
        if select_option == 'first':
            paths_to_datafiles = paths_to_datafiles[:N_data]
        else:
            paths_to_datafiles = random.sample(paths_to_datafiles, N_data)    
    

    return paths_to_datafiles


def get_data_by_instance_split_sl(instances_split_file, parent_instances_dir, parent_data_dir, 
                                  split='train', select_option='first', N_instances=float('inf')):

    if N_instances == 0 or not os.path.exists(instances_split_file):
        return []
        
    instances_split = load_instances_split(instances_split_file, split, float('inf'), select_option)
    if len(instances_split) == 0:
        return []
    
    paths_to_datafiles = []
    num_loaded = 0
    for path in instances_split:  
        basename = os.path.basename(path)
        instance_dir = os.path.dirname(path).replace(parent_instances_dir, '').replace(basename, '').lstrip('/') 
        instance_idx = replace_extension(basename)

        # single data file per instance
        data_files = get_gzip_files_by_extension(os.path.join(parent_data_dir, instance_dir), file_prefix=f'data_{instance_idx}')
        paths_to_datafiles.extend(data_files)
        if len(data_files) > 0:
            num_loaded += 1
        if num_loaded >= N_instances:
            break

    return paths_to_datafiles


def get_data_train_val_test_split(instances_split_file, parent_instances_dir, parent_data_dir, instances_dir, task='branching',
                             select_option='first', Ntrain_instances=float('inf'), Nval_instances=float('inf'), Ntest_instances=float('inf'),
                             max_data_per_instance=float('inf'), Ntrain_data=float('inf'), Nval_data=float('inf'), Ntest_data=float('inf')):
    
    # check that train split exists and update it if we miss any intance in the previous split
    get_instances_train_val_test_split(instances_split_file, parent_instances_dir, instances_dir)
    if task == 'branching':
        train_split = get_data_by_instance_split_branching(instances_split_file, parent_instances_dir, 
                                                           parent_data_dir, split='train',
                                                           select_option=select_option, N_instances=Ntrain_instances, 
                                                           max_data_per_instance=max_data_per_instance, N_data=Ntrain_data)
        val_split = get_data_by_instance_split_branching(instances_split_file, parent_instances_dir, 
                                                         parent_data_dir, split='val',
                                                         select_option=select_option, N_instances=Nval_instances, 
                                                         max_data_per_instance=max_data_per_instance, N_data=Nval_data)
        test_split = get_data_by_instance_split_branching(instances_split_file, parent_instances_dir,
                                                          parent_data_dir, split='test',
                                                          select_option=select_option, N_instances=Ntest_instances, 
                                                          max_data_per_instance=max_data_per_instance, N_data=Ntest_data)
    else:
        train_split = get_data_by_instance_split_sl(instances_split_file, parent_instances_dir,
                                                    parent_data_dir, split='train',
                                                    select_option=select_option, N_instances=Ntrain_instances)
        val_split = get_data_by_instance_split_sl(instances_split_file, parent_instances_dir,
                                                  parent_data_dir, split='val',
                                                  select_option=select_option, N_instances=Nval_instances)
        test_split = get_data_by_instance_split_sl(instances_split_file, parent_instances_dir,
                                                   parent_data_dir, split='test',
                                                   select_option=select_option, N_instances=Ntest_instances)
    return train_split, val_split, test_split


def get_data_train_val_test_split_list(instances_split_files, parent_instances_dir, parent_data_dir, instances_dir_list, task='branching',
                                  select_option='first', Ntrain_instances=float('inf'), Nval_instances=float('inf'), Ntest_instances=float('inf'),
                                  Ntrain_data=float('inf'), Nval_data=float('inf'), Ntest_data=float('inf'), max_data_per_instance=float('inf')):
    train_split, val_split, test_split = [], [], []
    for i_dir, (instances_dir, instances_split_file) in enumerate(zip(instances_dir_list, instances_split_files)):
        train_split_, val_split_, test_split_ = get_data_train_val_test_split(instances_split_file, parent_instances_dir, parent_data_dir, instances_dir, task=task, 
                                                                select_option=select_option, Ntrain_instances=Ntrain_instances, Nval_instances=Nval_instances, 
                                                                Ntest_instances=Ntest_instances, Ntrain_data=Ntrain_data, Nval_data=Nval_data, Ntest_data=Ntest_data, 
                                                                max_data_per_instance=max_data_per_instance)
        train_split.extend(train_split_)
        val_split.extend(val_split_)
        test_split.extend(test_split_)

        if i_dir % 10 == 0:
            print(f'Finished loading {i_dir+1} / {len(instances_dir_list)} instances dirs ...')

    return train_split, val_split, test_split
    

def get_data_train_val_test_split_list_by_dir(instances_dir_split_file, parent_instances_dir, parent_instances_metadata_dir, 
                                              parent_data_dir, instances_dir_list, task='branching',
                                              select_option='first', Ntrain_instances=float('inf'), Nval_instances=float('inf'), Ntest_instances=float('inf'), 
                                              Ntrain_data=float('inf'), Nval_data=float('inf'), Ntest_data=float('inf'), max_data_per_instance=float('inf'),
                                              select_splits=['train', 'val', 'test']):
    # collect instances from directories, collect a subset of train, val, test instances and merge them together as the new set
    train_instances_dir_split, val_instances_dir_split, test_instances_dir_split = get_instances_dir_train_val_test_split(instances_dir_split_file, instances_dir_list)
    if 'train' not in select_splits:
        train_instances_dir_split = []
    if 'val' not in select_splits:
        val_instances_dir_split = []
    if 'test' not in select_splits:
        test_instances_dir_split = []

    train_split, val_split, test_split = [], [], []

    n_classes_with_data = 0
    for split, instances_dir_split in [('train', train_instances_dir_split), ('val', val_instances_dir_split), ('test', test_instances_dir_split)]:
        print(f'{split}: {len(instances_dir_split)} to check::', instances_dir_split, '...')
        if len(instances_dir_split) == 0:
            print(f'No instance {split} split found. The user may need to generate a new split.')
            continue

        for i_dir, instances_dir in enumerate(instances_dir_split):  #  tqdm.tqdm(instances_dir_split):
            instances_dir = instances_dir.replace(parent_instances_dir, '').lstrip('/')
            instances_split_file = os.path.join(parent_instances_metadata_dir, instances_dir, f'instances_split.json')
            if 'samples' in instances_dir:  # generated by G2MILP/ACM-MILP baselines, take all instances and do not subsample
                train_split_, val_split_, test_split_ = get_data_train_val_test_split(instances_split_file, parent_instances_dir, parent_data_dir, instances_dir, task=task, 
                                                                                select_option=select_option, Ntrain_instances=float('inf'), Nval_instances=float('inf'), 
                                                                                Ntest_instances=float('inf'), Ntrain_data=Ntrain_data, Nval_data=Nval_data, Ntest_data=Ntest_data, 
                                                                                max_data_per_instance=max_data_per_instance)
            else:
                train_split_, val_split_, test_split_ = get_data_train_val_test_split(instances_split_file, parent_instances_dir, parent_data_dir, instances_dir, task=task, 
                                                                                    select_option=select_option, Ntrain_instances=Ntrain_instances, Nval_instances=Nval_instances, 
                                                                                    Ntest_instances=Ntest_instances, Ntrain_data=Ntrain_data, Nval_data=Nval_data, Ntest_data=Ntest_data, 
                                                                                    max_data_per_instance=max_data_per_instance)
            data_split_ = train_split_ + val_split_ + test_split_ if split != 'val' else val_split_
            if split == 'train':
                train_split.extend(data_split_) 
            elif split == 'val':
                val_split.extend(data_split_) 
            else:
                test_split.extend(data_split_)
            if len(data_split_) > 0:
                n_classes_with_data += 1

            if i_dir % 10 == 0:
                print(f'Finished loading {i_dir+1} / {len(instances_dir_split)} instances dirs ...')
    print(f'--------------- # Classes with data {n_classes_with_data}... ------------------')
    return train_split, val_split, test_split



def get_stats(text, last_n=10):
    lines = text.split('\n')[-last_n:]
    solve_status = None
    solve_time = None
    primal_bound = None
    dual_bound = None
    gap = None

    primal_pattern = r"Primal Bound\s+:\s+([+\-]?\d+\.\d+e[+\-]?\d+)"
    dual_pattern = r"Dual Bound\s+:\s+([+\-]?\d+\.\d+e[+\-]?\d+)"

    for line in lines:
        if "Solve Status:" in line:
            solve_status_match = re.search(r"Solve Status:\s*(.*)", line)
            if solve_status_match:
                solve_status = solve_status_match.group(1).strip()
        elif "Solve Time:" in line:
            solve_time_match = re.search(r"Solve Time:\s*([\d.]+)\s*seconds", line)
            if solve_time_match:
                solve_time = float(solve_time_match.group(1))
        
        elif "Gap" in line:
            # Gap                : 0.00 %
            # Gap                : infinite
            gap_match = re.search(r"Gap\s*:\s*([\d.]+\s*)%", line)
            if gap_match:
                gap = float(gap_match.group(1))
        else:
            # primal or dual bound
            primal_match = re.search(primal_pattern, line)
            dual_match = re.search(dual_pattern, line)
            if primal_match:
                # Primal Bound       : +1.70000000000000e+01 (95 solutions)
                primal_bound = float(primal_match.group(1))
            elif dual_match:
                # Dual Bound         : +8.31117561544689e+01
                dual_bound = float(dual_match.group(1))
    return solve_status, solve_time, primal_bound, dual_bound, gap

        
    
