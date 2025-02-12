import json
import pickle
import random
import re
import time
from typing import AnyStr, List, Union
import gap_data as _data
import torch
import torch_geometric
from torch_geometric.data import Data
from contrast_class_split import milp_id
from utils import load_gzip


class MyDataWithLabelsAndCodeText(Data):
    ###  
    row_dim = _data.ROW_DIM
    edge_dim_rowcols = _data.EDGE_DIM_ROWCOLS
    col_dim = _data.COL_DIM
    ###

    def __init__(
            self,
            x_rows,
            x_cols,
            edge_index_rowcols,
            edge_vals_rowcols,
            label,
            code_text):  #  

        super().__init__()
        self.x_rows = x_rows
        self.x_cols = x_cols
        self.edge_index_rowcols = edge_index_rowcols
        self.edge_vals_rowcols = edge_vals_rowcols
        self.label = label
        self.code_text = code_text

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'x_rows':
            inc = 0
        elif key == 'x_cols':
            inc = 0
        elif key == 'edge_index_rowcols':
            inc = torch.tensor([
                [self.x_rows.size(0)],
                [self.x_cols.size(0)]])
        elif key == 'edge_vals_rowcols':
            inc = 0
        elif key == 'label':
            inc = 0
        elif key == 'code_text':
            inc = 0
        else:
            print(f'{key}:: Resorting to default')
            inc = super().__inc__(key, value, *args, **kwargs)
        return inc

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'x_rows':
            cat_dim = 0
        elif key == 'x_cols':
            cat_dim = 0
        elif key == 'edge_index_rowcols':
            cat_dim = 1
        elif key == 'edge_vals_rowcols':
            cat_dim = 0
        elif key == 'label':
            cat_dim = 0
        elif key == 'code_text':
            cat_dim = 0
        else:
            print(f'{key}:: Resorting to default')
            cat_dim = super().__cat_dim__(key, value, *args, **kwargs)
        return cat_dim
    


class my_mps_text_dataset(torch_geometric.data.Dataset):
    def __init__(self, filenames: Union[List, AnyStr], num_milp_instance:int=None, num_milp_class=None, text_types:str="all"):
        """
        Initialize the MPS text dataset.

        Args:
            filenames (Union[List, AnyStr]): Path(s) to the dataset file(s). Can be either JSON or gzipped pickle files.
            num_milp (int, optional): Maximum number of MILPs to include in the dataset. 
                                      If the dataset contains more MILPs than this number, 
                                      a random subset will be selected. Defaults to 1e9.
                                      If None, then we don't filter the milp's name; it needs
                                      to be None for miplib, seed, and datasets without milp ID.
            text_types (str, optional): Type of text to include. Can be "all" or "description only". 
                                        Defaults to "all".

        The dataset is loaded from the file(s) and processed according to the num_milp parameter.
        Each data point in the dataset corresponds to an MPS file and associated code texts.
        """
        super().__init__(root=None, transform=None, pre_transform=None)
        if isinstance(filenames, str):
            # cast to 1D list if it is string.
            filenames = [filenames, ]

        self.data = {}
        for filename in filenames:
            if filename.endswith('.json'):
                self.data.update(json.load(open(filename, 'r')))
            elif filename.endswith('.pkl.gz'):
                self.data.update(pickle.load(open(filename, 'rb')))

        if num_milp_instance or num_milp_class:
            # NOTE: usually we only use one of the two filters. Now, if both filters are set
            # we first filter the milp class and then the milp instance.
            milp_ids = [milp_id(path) for path in self.data.keys()]
            milp_ids = list(set(milp_ids))
            print("MAX MILP ID", max(milp_ids))
            if num_milp_class and len(milp_ids) > num_milp_class:
                # subsample the milp class
                random.seed(time.time())
                print(f"Selecting {num_milp_class} MILPs classes from {len(milp_ids)} MILP classes")
                milp_ids = random.sample(milp_ids, num_milp_class)
                self.data = {path: self.data[path] for path in self.data.keys() if milp_id(path) in milp_ids}

            if num_milp_instance and len(self.data) > num_milp_instance:
                random.seed(time.time())
                print(f"Selecting {num_milp_instance} instances from {len(self.data)} instances")
                self.data = dict(random.sample(self.data.items(), num_milp_instance))

        if text_types == "description only":
            # we know the first item in the array is always the description.
            self.data =  {path: [arr[0]] for path, arr in self.data.items()} 
        else:
            assert text_types == "all"

        self.mps_files = list(self.data.keys())
        self.mps_files = sorted(self.mps_files)
        
    def len(self):
        return len(self.mps_files)
    
    def get(self, index):
        train_data = load_gzip(self.mps_files[index])
        code_texts = self.data[self.mps_files[index]]
        code_text = random.choice(code_texts)

        if isinstance(train_data, tuple) or isinstance(train_data, list):
            # (data, label_dict)
            train_data = train_data[0]

        return MyDataWithLabelsAndCodeText(
            train_data.x_rows,
            train_data.x_cols,
            train_data.edge_index_rowcols,
            train_data.edge_vals_rowcols,
            None,
            code_text
        )



def getDataloadersWithCodeTexts(filename: Union[List, AnyStr], num_milp_instance:int=None, num_milp_class:int=None,
        batch_size:int=64, shuffle_flag:bool=True, pin_memory:bool=False, text_types:str="all"):
    follow_batch = ['x_rows', 'x_cols']

    trainloader = torch_geometric.loader.DataLoader(
        my_mps_text_dataset(filename, num_milp_instance=num_milp_instance, num_milp_class=num_milp_class, text_types=text_types), 
        batch_size=batch_size, 
        shuffle=shuffle_flag,
        follow_batch=follow_batch, 
        num_workers=0, 
        pin_memory=pin_memory
    )

    return trainloader