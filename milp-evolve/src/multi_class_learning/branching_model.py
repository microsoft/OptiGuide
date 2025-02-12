import gzip
import pickle
from pathlib import Path

import ecole
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import TransformerConv
from transformers import AutoModel, AutoTokenizer


class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size=64, edge_nfeats=1):
        super().__init__("add")
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(edge_nfeats, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output
    

class GNNPolicy(torch.nn.Module):
    def __init__(self, emb_size=64, cons_nfeats=5, edge_nfeats=1, var_nfeats=19, output_size=1,
                 n_layers=1):
        super().__init__()
        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.n_layers = n_layers
        
        if n_layers == 1:
            self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats)
            self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats)
        else:
            for i in range(n_layers):
                setattr(self, f"conv_{i}_v_to_c", BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats))
                setattr(self, f"conv_{i}_c_to_v", BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats))
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, output_size, bias=False),
        )

    def forward_embed(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        if self.n_layers == 1:
            constraint_features = self.conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )
        else:
            for i in range(self.n_layers):
                conv_v_to_c = getattr(self, f"conv_{i}_v_to_c")
                conv_c_to_v = getattr(self, f"conv_{i}_c_to_v")
                constraint_features = constraint_features + conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                )
                variable_features = variable_features + conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )

        return variable_features, constraint_features


    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        if self.n_layers == 1:
            constraint_features = self.conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )
        else:
            for i in range(self.n_layers):
                conv_v_to_c = getattr(self, f"conv_{i}_v_to_c")
                conv_c_to_v = getattr(self, f"conv_{i}_c_to_v")
                constraint_features = constraint_features + conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                )
                variable_features = variable_features + conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
