
import gzip
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from utils import load_gzip, load_json


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        candidates,
        nb_candidates,
        candidate_choice,
        candidate_scores,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, edge_nfeats=2):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.edge_nfeats = edge_nfeats

    def len(self):
        return len(self.sample_files)

    def get(self, index, nan_mask_val=0):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        sample = load_gzip(self.sample_files[index])
        sample_observation, sample_action, sample_action_set, sample_scores = sample
        
        constraint_features = sample_observation.row_features
        edge_indices = sample_observation.edge_features.indices.astype(np.int32)
        edge_features = np.expand_dims(sample_observation.edge_features.values, axis=-1)
        if self.edge_nfeats == 2:
            edge_features_norm = edge_features / np.linalg.norm(edge_features) 
            edge_features = np.concatenate((edge_features, edge_features_norm), axis=-1)
        variable_features = sample_observation.variable_features

        constraint_features = np.nan_to_num(constraint_features, nan=nan_mask_val)
        edge_features = np.nan_to_num(edge_features, nan=nan_mask_val)
        variable_features = np.nan_to_num(variable_features, nan=nan_mask_val)

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        candidates = np.array(sample_action_set, dtype=np.int32)
        candidate_scores = np.array([sample_scores[j] for j in candidates])
        candidate_choice = np.where(candidates == sample_action)[0][0]

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(candidates),
            len(candidates),
            torch.LongTensor([candidate_choice]),
            torch.FloatTensor(candidate_scores)
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph
