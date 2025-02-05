import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx


class ProteinFolding:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        acids = range(self.n_acid)
        n_h_phobic = int(self.ratio * self.n_acid)
        h_phobic = random.sample(acids, n_h_phobic)

        graph = nx.barabasi_albert_graph(self.n_acid, self.m)
        adj_matrix = nx.to_numpy_array(graph)

        ans = {'acids': acids, 'h_phobic': h_phobic, 'adj_matrix': adj_matrix}
        return ans

    def solve(self, instance):
        acids = instance['acids']
        h_phobic = instance['h_phobic']
        adj_matrix = instance['adj_matrix']

        list_ij = [(i, j) for i in h_phobic for j in h_phobic if j > i + 1]

        list_ik1j = []
        list_ik2j = []

        for i, j in list_ij:
            for k in range(i, j):
                if k == (i + j - 1) / 2:
                    list_ik2j.append((i, j, k))
                else:
                    list_ik1j.append((i, j, k))

        ijfold = [(i, j) for i, j, _ in list_ik2j]

        model = Model('ProteinFolding')

        match = {(i, j): model.addVar(vtype="B", name=f"match_{i}_{j}") for i, j in list_ij}
        fold = {k: model.addVar(vtype="B", name=f"fold_{k}") for k in acids}
        adj_fold = {(i, j): model.addVar(vtype="B", name=f"adj_fold_{i}_{j}") for i in acids for j in acids if adj_matrix[i, j] == 1}

        for i, j, k in list_ik1j:
            model.addCons(fold[k] + match[i, j] <= 1, f"C1_{i}_{j}_{k}")

        for i, j, k in list_ik2j:
            model.addCons(match[i, j] <= fold[k], f"C2_{i}_{j}_{k}")

        for i in range(len(acids)):
            for j in range(len(acids)):
                if adj_matrix[i][j] == 1:
                    model.addCons(fold[i] + fold[j] <= 1 + adj_fold[i, j], f"C3_adj_fold_{i}_{j}")

        objective_expr = quicksum(match[i, j] for i, j in ijfold) + 0.5 * quicksum(adj_fold[i, j] for i in acids for j in acids if adj_matrix[i, j] == 1)

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_acid': 80,
        'ratio': 0.4,
        'm': 2,
    }

    protein_folding = ProteinFolding(parameters, seed=seed)
    instance = protein_folding.generate_instance()
    solve_status, solve_time = protein_folding.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")