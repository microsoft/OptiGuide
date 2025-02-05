import torch
from torch_geometric.data import Data
import numpy as np
import os

import utils as _utils

ROW_DIM = 29  
EDGE_DIM_ROWCOLS = 2
COL_DIM = 17  


# Dataset with both data and label
class MyDataWithLabels(Data):
    ###  
    row_dim = ROW_DIM
    edge_dim_rowcols = EDGE_DIM_ROWCOLS
    col_dim = COL_DIM
    ###

    def __init__(
            self,
            x_rows,
            x_cols,
            edge_index_rowcols,
            edge_vals_rowcols,
            label):  #  

        super().__init__()
        self.x_rows = x_rows
        self.x_cols = x_cols
        self.edge_index_rowcols = edge_index_rowcols
        self.edge_vals_rowcols = edge_vals_rowcols
        self.label = label

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
        else:
            print(f'{key}:: Resorting to default')
            cat_dim = super().__cat_dim__(key, value, *args, **kwargs)
        return cat_dim



class MyData(Data):
    ###  
    row_dim = ROW_DIM
    edge_dim_rowcols = EDGE_DIM_ROWCOLS
    col_dim = COL_DIM
    ###

    def __init__(
            self,
            x_rows,
            x_cols,
            edge_index_rowcols,
            edge_vals_rowcols):  #  

        super().__init__()
        self.x_rows = x_rows
        self.x_cols = x_cols
        self.edge_index_rowcols = edge_index_rowcols
        self.edge_vals_rowcols = edge_vals_rowcols

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
        else:
            print(f'{key}:: Resorting to default')
            cat_dim = super().__cat_dim__(key, value, *args, **kwargs)
        return cat_dim

    @classmethod
    def from_path(cls, path):
        raw_data = cls.load_rawdata(path)
        maxnum_rowcolixs = cls.get_maxnums(path)
        processed_data = cls.from_rawdata(raw_data, maxnum_rowcolixs)
        self = cls(*processed_data)
        return self

    @classmethod
    def get_maxnums(cls, path):
        # for cut <-> col
        maxnum_rowcolixs = 200 if 'nn' in path else 1e6
        return maxnum_rowcolixs
        
    @classmethod
    def load_rawdata(cls, path):
        raw_data = {}
        for file in ['row_input_scores.npy']:  #  

            arr = _utils.load_numpy(os.path.join(path, file))
            raw_data[file] = arr

        for file in ['row_features.pkl',
                     'col_features.pkl',
                     'row_coefs.pkl']:  #  

            feat_dicts = _utils.load_pickle(os.path.join(path, file))
            raw_data[file] = feat_dicts

        return raw_data

    @classmethod
    def from_rawdata(cls, raw_data, maxnum_rowcolixs=1e6):
        x_rows = cls.make_xrows(
            raw_data['row_features.pkl'],
            raw_data['row_input_scores.npy'],
        )

        x_cols = cls.make_xcols(raw_data['col_features.pkl'])

        edge_index_rowcols, edge_vals_rowcols = cls.make_edge_cols(
            raw_data['row_coefs.pkl'], maxnum_rowcolixs
        )

        if torch.any(torch.isinf(x_rows)):
            x_rows[torch.isinf(x_rows)] = 0
            print('Certain features are inf!!')

        return (x_rows, x_cols, edge_index_rowcols, edge_vals_rowcols)

    @classmethod
    def make_xrows(cls, features, scores):
        vecs = []
        for feat_dict in features:
            vec = cls.get_row_vec(feat_dict)
            vecs.append(vec)

        vecs = np.array(vecs)

        try:
            scores[:, -8] = np.clip( (scores[:, -8] - np.mean(scores[:, -8])) / (np.std(scores[:, -8]) + 1e-5), a_min = -10, a_max=10)
            scores[:, -3] = np.clip( (scores[:, -3] - np.mean(scores[:, -3])) / (np.std(scores[:, -3]) + 1e-5), a_min = -10, a_max=10)
        except:
            import pdb; pdb.set_trace()

        x = torch.FloatTensor(np.concatenate([vecs, scores], axis=-1))
        # this feature is inf for one row often.
        x[:, 22][torch.isinf(x[:, 22])] = -1.0  
        return x

    @classmethod
    def make_xcols(cls, features):
        vecs = []
        for feat_dict in features:
            vec = cls.get_col_vec(feat_dict)
            vecs.append(vec)

        x = torch.FloatTensor(np.array(vecs))

        return x

    @classmethod
    def make_edge_cols(cls, coefs, maxnum=1e6):
        edge_ixs_top = []
        edge_ixs_bottom = []
        edge_vals_raw = []
        edge_vals_norm = []

        for cut_ix, (col_ix, vals) in coefs.items():
            edge_ixs_top.append( np.ones(len(col_ix)) * cut_ix )
            edge_ixs_bottom.append( np.array(col_ix) )

            vals = np.array(vals)
            edge_vals_raw.append( vals )
            edge_vals_norm.append( vals / np.linalg.norm(vals))

        if len(edge_ixs_top) == 0:  #  
            edge_ixs = np.stack([[], []])
            edge_vals = np.stack([[], []])
        else:
            edge_ixs = np.stack([
                np.concatenate(edge_ixs_top),
                np.concatenate(edge_ixs_bottom)])
            edge_vals = np.stack([
                np.concatenate(edge_vals_raw),
                np.concatenate(edge_vals_norm)])

        edge_ixs = torch.LongTensor(edge_ixs)
        edge_vals = torch.FloatTensor(edge_vals.T)

        # Filter..
        if maxnum < 1e6:  # nnv
            if len(edge_vals) > maxnum:
                threshold = torch.sort(edge_vals[:, 0], descending=True).values[maxnum]
                mask = (edge_vals[:, 0] > threshold)
                edge_vals = torch.stack( [
                    torch.masked_select(edge_vals[:,0], mask),
                    torch.masked_select(edge_vals[:,1], mask),
                    ], dim=-1)
                edge_ixs = torch.stack( [
                    torch.masked_select(edge_ixs[0, :], mask),
                    torch.masked_select(edge_ixs[1, :], mask),
                    ], dim=0)
            else:
                pass

        return edge_ixs, edge_vals

   
    @classmethod
    def get_row_vec(cls, feat_dict):
        vec = []
        # Type
        vec.append(1.0 if feat_dict['origin_type'] == 0 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 1 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 2 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 3 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 4 else 0.0 )

        # basis status
        vec.append( 1.0 if feat_dict['basisstatus'] == 0 else 0.0 )  # basestat one-hot {lower: 0, basic: 1, upper: 2, zero: 3}
        vec.append( 1.0 if feat_dict['basisstatus'] == 1 else 0.0 )
        vec.append( 1.0 if feat_dict['basisstatus'] == 2 else 0.0 )
        vec.append( 1.0 if feat_dict['basisstatus'] == 3 else 0.0 )
        # other info
        vec.append( feat_dict['rank'] )
        # normalization (following Giulia)
        lhs = feat_dict['lhs']
        rhs = feat_dict['rhs']
        cst = feat_dict['cst']
        nlps = feat_dict['nlps']
        cste = feat_dict['cste']

        activity = feat_dict['activity']
        row_norm = feat_dict['row_norm']
        obj_norm = feat_dict['obj_norm']
        dualsol = feat_dict['dualsol']

        unshifted_lhs = None if np.isinf(lhs) else lhs - cst
        unshifted_rhs = None if np.isinf(rhs) else rhs - cst

        if unshifted_lhs is not None:
            bias = -1. * unshifted_lhs / row_norm
            dualsol = -1. *  dualsol / (row_norm * obj_norm)
        if unshifted_rhs is not None:
            bias = unshifted_rhs / row_norm
            dualsol = dualsol / (row_norm * obj_norm)
        # values
        vec.append( bias )
        vec.append( dualsol )
        vec.append( 1.0 if np.isclose(activity, lhs) else 0.0 ) # at_lhs
        vec.append( 1.0 if np.isclose(activity, rhs) else 0.0 ) # at_rhs
        vec.append( feat_dict['nlpnonz'] / feat_dict['ncols'] )
        vec.append( feat_dict['age'] / (feat_dict['nlps'] + feat_dict['cste']) )
        vec.append( feat_dict['nlpsaftercreation'] / (feat_dict['nlps'] + feat_dict['cste']) )
        vec.append( feat_dict['intcols'] / feat_dict['ncols'] )

        # flags
        vec.append( 1.0 if feat_dict['is_integral'] else 0.0 )
        vec.append( 1.0 if feat_dict['is_removable'] else 0.0 )  # could be removed if we have binary identifier for cuts
        vec.append( 1.0 if feat_dict['is_in_lp'] else 0.0 )  # could be removed if we have binary identifier for cuts

        return vec

    @classmethod
    def get_col_vec(cls, feat_dict):
        vec = []
        # type
        vec.append( 1.0 if feat_dict['type'] == 0 else 0.0 ) # binary
        vec.append( 1.0 if feat_dict['type'] == 1 else 0.0 ) # integer
        vec.append( 1.0 if feat_dict['type'] == 2 else 0.0 ) # implicit-int
        vec.append( 1.0 if feat_dict['type'] == 3 else 0.0 ) # continuous
        # bounds
        vec.append( 1.0 if feat_dict['lb'] is not None else 0.0 ) # has_lower_bound
        vec.append( 1.0 if feat_dict['ub'] is not None else 0.0 ) # has_upper_bound
        # basis status
        vec.append( 1.0 if feat_dict['basestat'] == 0 else 0.0 )
        vec.append( 1.0 if feat_dict['basestat'] == 1 else 0.0 )
        vec.append( 1.0 if feat_dict['basestat'] == 2 else 0.0 )
        vec.append( 1.0 if feat_dict['basestat'] == 3 else 0.0 )
        # values
        vec.append( feat_dict['norm_coef'])
        vec.append( feat_dict['norm_redcost'] ) # was already normalized :()
        vec.append( feat_dict['norm_age'] ) # war already normalized :()
        vec.append( feat_dict['solval'])
        vec.append( feat_dict['solfrac'])
        vec.append( 1.0 if feat_dict['sol_is_at_lb'] else 0.0 )
        vec.append( 1.0 if feat_dict['sol_is_at_ub'] else 0.0 )

        return vec
