import torch_geometric
import gap_data as _data
import os
from utils import load_gzip


class my_dataset(torch_geometric.data.Dataset):
    def __init__(self, train_files, label_key='lp_ip_gap'):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.train_files = [train_file for train_file in train_files if os.path.exists(train_file)]
        self.label_key = label_key
    
    def len(self):
        return len(self.train_files)
    
    def get(self, index):
        train_data = load_gzip(self.train_files[index])
        label = train_data[1]
        if isinstance(label, dict):
            label = label[self.label_key]
        
        return _data.MyDataWithLabels(
            train_data[0].x_rows,
            train_data[0].x_cols,
            train_data[0].edge_index_rowcols,
            train_data[0].edge_vals_rowcols,
            label
        )


def getDataloaders(train_files, batch_size=64, shuffle_flag=True, pin_memory=False, label_key='lp_ip_gap'):
    follow_batch = ['x_rows', 'x_cols']

    trainloader = torch_geometric.loader.DataLoader(
        my_dataset(train_files, label_key=label_key), 
        batch_size=batch_size, 
        shuffle=shuffle_flag,
        follow_batch=follow_batch, 
        num_workers=0, 
        pin_memory=pin_memory
    )

    return trainloader




################ For LLava, no labels ################
class my_dataset_no_labels(torch_geometric.data.Dataset):
    def __init__(self, train_files, label_key='lp_ip_gap'):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.train_files = [train_file for train_file in train_files if os.path.exists(train_file)]
        self.label_key = label_key
    
    def len(self):
        return len(self.train_files)
    
    def get(self, index):
        train_data = load_gzip(self.train_files[index])
        return _data.MyData(
            train_data.x_rows,
            train_data.x_cols,
            train_data.edge_index_rowcols,
            train_data.edge_vals_rowcols
        )


def getDataloadersNoLabels(train_files, batch_size=64, shuffle_flag=True, pin_memory=False, label_key='lp_ip_gap'):
    follow_batch = ['x_rows', 'x_cols']

    trainloader = torch_geometric.loader.DataLoader(
        my_dataset_no_labels(train_files, label_key=label_key), 
        batch_size=batch_size, 
        shuffle=shuffle_flag,
        follow_batch=follow_batch, 
        num_workers=0, 
        pin_memory=pin_memory
    )

    return trainloader


class my_dataset_no_labels_from_data(torch_geometric.data.Dataset):
    def __init__(self, train_data, label_key='lp_ip_gap'):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.train_data = train_data
        self.label_key = label_key
    
    def len(self):
        return len(self.train_data)
    
    def get(self, index):
        train_data = self.train_data[index]
        return _data.MyData(
            train_data.x_rows,
            train_data.x_cols,
            train_data.edge_index_rowcols,
            train_data.edge_vals_rowcols
        )


def getDataloadersNoLabelsFromData(train_data, batch_size=64, shuffle_flag=True, pin_memory=False, label_key='lp_ip_gap'):
    follow_batch = ['x_rows', 'x_cols']

    trainloader = torch_geometric.loader.DataLoader(
        my_dataset_no_labels_from_data(train_data, label_key=label_key), 
        batch_size=batch_size, 
        shuffle=shuffle_flag,
        follow_batch=follow_batch, 
        num_workers=0, 
        pin_memory=pin_memory
    )

    return trainloader




