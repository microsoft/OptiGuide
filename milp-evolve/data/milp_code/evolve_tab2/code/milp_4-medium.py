import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum


class BinPacking:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity) 
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        A = np.random.randint(5, 30, size=(self.n_bins, self.n_items))
        b = np.random.randint(10 * self.n_items, 15 * self.n_items, size=self.n_bins)
        c = np.random.randint(1, 20, size=self.n_items)

        res = {'A': A, 'b': b, 'c': c}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        A = instance['A']
        b = instance['b']
        c = instance['c']
        
        # build the optimization model
        model = Model("BinPacking")

        x = {}
        for j in range(self.n_items):
            x[j] = model.addVar(vtype='B', lb=0.0, ub=1, name="x_%s" % (j+1))
        
        for i in range(self.n_bins):
            model.addCons(quicksum(A[i, j]*x[j] for j in range(self.n_items)) <= b[i], "Resource_%s" % (i+1))

        objective_expr = quicksum(c[j]*x[j] for j in range(self.n_items))

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_bins': 400,
        'n_items': 400,
    }

    bin_packing = BinPacking(parameters, seed=seed)
    instance = bin_packing.generate_instance()
    solve_status, solve_time = bin_packing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
