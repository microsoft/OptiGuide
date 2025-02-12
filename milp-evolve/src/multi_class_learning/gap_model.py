import gap_data as _data
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv


# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """
    def __init__(self, emb_size=64, edge_dim=2, do_gat=False, heads=8, dropout=0.6):
        super().__init__('add')
        self.emb_size = emb_size

        self.feature_module_left = torch.nn.Sequential(
            # torch.nn.LayerNorm(emb_size),
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            # torch.nn.LayerNorm(emb_size),
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )

        self.do_gat = do_gat
        if do_gat:
            self.gat_conv = GATConv(emb_size, emb_size, heads=heads, edge_dim=edge_dim, concat=True, dropout=dropout)

        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        if self.do_gat:
            output = self.feature_module_final(self.gat_conv(node_features_i, node_features_j, edge_features))
        else:
            output = self.feature_module_final(self.feature_module_left(node_features_i)
                                            + self.feature_module_edge(edge_features)
                                            + self.feature_module_right(node_features_j))
        return output



class MyGNNAttn(torch.nn.Module):
    row_dim = _data.MyData.row_dim
    col_dim = _data.MyData.col_dim
    edge_dim_rowcols = _data.MyData.edge_dim_rowcols
 
    def __init__(self, emb_size=64, n_gnn_iters=1, n_out_neurons=1, edge_nfeats=2, norm_method='layernorm',
                 do_gat=False, heads=8, dropout=0.6, n_attn_iters=1, max_token_attn=512):
        super().__init__()
        self.emb_size = emb_size
        self.n_gnn_iters = n_gnn_iters
        self.n_out_neurons = n_out_neurons
        self.edge_nfeats = edge_nfeats
        self.max_token_attn = max_token_attn
        # ROW EMBEDDING
        norm_fn = torch.nn.BatchNorm1d if norm_method == 'batchnorm' else torch.nn.LayerNorm
        self.row_embedding = torch.nn.Sequential(
            norm_fn(self.row_dim),
            torch.nn.Linear(self.row_dim, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
 
        # Column EMBEDDING
        self.col_embedding = torch.nn.Sequential(
            norm_fn(self.col_dim),
            torch.nn.Linear(self.col_dim, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
 
        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            norm_fn(edge_nfeats),
        )
 
        if self.n_gnn_iters == 1:
            self.conv_col_to_row = BipartiteGraphConvolution(emb_size=emb_size, edge_dim=edge_nfeats,
                                                            do_gat=do_gat, heads=heads, dropout=dropout)
            self.conv_row_to_col = BipartiteGraphConvolution(emb_size=emb_size, edge_dim=edge_nfeats,
                                                            do_gat=do_gat, heads=heads, dropout=dropout)
        else:
            for i in range(self.n_gnn_iters):
                setattr(self, f'conv_col_to_row_{i}', BipartiteGraphConvolution(emb_size=emb_size, edge_dim=edge_nfeats,
                                                            do_gat=do_gat, heads=heads, dropout=dropout))
                # self[f"conv_col_to_row_{i}_LN"] = nn.LayerNorm(emb_size)
                setattr(self, f'conv_row_to_col_{i}', BipartiteGraphConvolution(emb_size=emb_size, edge_dim=edge_nfeats,
                                                            do_gat=do_gat, heads=heads, dropout=dropout))
 
        self.col_output_embed_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
       
        self.row_output_embed_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
       
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, n_out_neurons),
        )
 
        # add 3 self-attention layers
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=8, batch_first=True)
        self.self_attn = torch.nn.TransformerEncoder(enc_layer, num_layers=n_attn_iters, norm=nn.LayerNorm(emb_size))
       
        # Apply the initialization function to the model
        self.apply(init_weights)
 
    def _get_device(self):
        return next(self.parameters()).device
 
    def _get_autocast_dtype(self):
        param_dtype = next(self.parameters()).dtype
        if param_dtype == torch.float32:
            return torch.float32
        elif param_dtype == torch.bfloat16:
            return torch.bfloat16
        elif param_dtype == torch.float16:
            return torch.float16
        elif param_dtype == torch.uint8:
            return torch.uint8
        elif param_dtype == torch.int4:
            raise ValueError("4-bit precision is not natively supported by PyTorch. Consider using a third-party library.")
        else:
            raise ValueError(f"Unsupported precision type: {param_dtype}")
       
    def get_embedding(self, inp, do_precision_cast=True):
        if do_precision_cast and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=self._get_autocast_dtype()):
                return self._get_embedding_impl(inp)
        else:
            return self._get_embedding_impl(inp)
 
    def process_input(self, x_rows, x_cols, edge_vals_rowcols):
        # isnan and inf to 0
        x_rows[torch.isnan(x_rows)] = 0
        x_rows[torch.isinf(x_rows)] = 0
        x_cols[torch.isnan(x_cols)] = 0
        x_cols[torch.isinf(x_cols)] = 0
        edge_vals_rowcols[torch.isnan(edge_vals_rowcols)] = 0
        edge_vals_rowcols[torch.isinf(edge_vals_rowcols)] = 0
        return x_rows, x_cols, edge_vals_rowcols
 
       
    def _get_embedding_impl(self, inp):
        #### Inputs to device
        device = self._get_device()
        dtype = self._get_autocast_dtype()
 
        x_rows = inp.x_rows.to(device, dtype) # constraints
        x_cols = inp.x_cols.to(device, dtype) # variables
 
        edge_index_rowcols = inp.edge_index_rowcols.to(device)  # row ix is top, col ix is bottom
        edge_vals_rowcols = inp.edge_vals_rowcols.to(device, dtype)  # currently fully connected, all 1 weight
       
        # for x_rows, x_cols and edge_vals_rowcols, map inf and nan to 0
        x_rows, x_cols, edge_vals_rowcols = self.process_input(x_rows, x_cols, edge_vals_rowcols)
 
        if self.edge_nfeats < edge_vals_rowcols.shape[1]:
            edge_vals_rowcols = edge_vals_rowcols[:, :self.edge_nfeats]
 
        if hasattr(inp, 'x_cols_batch'):
            x_cols_batch = inp.x_cols_batch.to(device)
            batch_size = inp.num_graphs
        else:
            x_cols_batch = torch.zeros(x_cols.shape[0], dtype=torch.long).to(device)
            batch_size = 1
 
        if hasattr(inp, 'x_rows_batch'):
            x_rows_batch = inp.x_rows_batch.to(device)
        else:
            x_rows_batch = torch.zeros(x_rows.shape[0], dtype=torch.long).to(device)
 
        r_edge_index_rowcols = torch.stack([edge_index_rowcols[1], edge_index_rowcols[0]], dim=0)
 
        row_embd = self.row_embedding(x_rows)
        col_embd = self.col_embedding(x_cols)
        edge_embd_rowcols = self.edge_embedding(edge_vals_rowcols)
 
        if self.n_gnn_iters == 1:
            row_embd = self.conv_col_to_row(col_embd, r_edge_index_rowcols, edge_embd_rowcols, row_embd)
            col_embd = self.conv_row_to_col(row_embd, edge_index_rowcols, edge_embd_rowcols, col_embd)
        else:
            for i in range(self.n_gnn_iters):
                row_embd = row_embd + getattr(self, f'conv_col_to_row_{i}')(col_embd, r_edge_index_rowcols, edge_embd_rowcols, row_embd)
                col_embd = col_embd + getattr(self, f'conv_row_to_col_{i}')(row_embd, edge_index_rowcols, edge_embd_rowcols, col_embd)
 
        row_embd = self.row_output_embed_module(row_embd)  # (nodes of constraints, feature)
        col_embd = self.col_output_embed_module(col_embd)  # (nodes of variables, feature)
 
 
        return self.special_attn_pooling_v2(row_embd, col_embd, x_rows_batch, x_cols_batch) # [bs, emb_size]
 
 
    def special_attn_pooling_v2(self, row_embd, col_embd, x_rows_batch, x_cols_batch):
        batch_size = x_rows_batch.max() + 1
        output = []
        for item_id in range(batch_size):
            row_mask = (x_rows_batch == item_id)
            row_tokens = row_embd[row_mask] # [num_tokens, emb_size]
            means_row = torch.mean(row_tokens, dim=0).reshape(1, -1) * 10.0
 
            col_mask = (x_cols_batch == item_id)
            col_tokens = col_embd[col_mask] + 0.5
            means_col = torch.mean(col_tokens, dim=0).reshape(1, -1) * 10.0
 
            tokens = torch.cat([row_tokens, col_tokens], dim=0)
 
            if tokens.shape[0] > self.max_token_attn:
                # randomly subsample
                indices = torch.randperm(tokens.shape[0])[:self.max_token_attn]
                tokens = tokens[indices]
               
            agg_tokens = torch.cat([tokens, means_row, means_col, torch.zeros(1, row_embd.shape[1]).to(row_embd.device)], dim=0)
            agg_tokens = agg_tokens.unsqueeze(0) # [1, num_nodes + mean token + special token, emb_size]
            embed = self.self_attn(agg_tokens)
            output.append(embed[0, -1, :])
 
        return torch.stack(output, dim=0) # [bs, emb_size]
 
       
    def forward(self, inp, do_precision_cast=True):
        if do_precision_cast and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=self._get_autocast_dtype()):
                return self._forward_impl(inp)
        else:
            return self._forward_impl(inp)
 
    def _forward_impl(self, inp):
        output = self._get_embedding_impl(inp)
        output = self.output_module(output)
 
        return output
    

# Define the initialization function
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)