import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimpleWarehouseStockingProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_random_warehouse(self):
        n_items = np.random.randint(self.min_items, self.max_items)
        W = nx.erdos_renyi_graph(n=n_items, p=self.incompat_prob, seed=self.seed)
        return W

    def generate_revenues(self, W):
        for item in W.nodes:
            W.nodes[item]['revenue'] = np.random.randint(1, 100)
    
    def generate_incompatibility(self, W):
        incompatible_pairs = set()
        for edge in W.edges:
            if np.random.random() <= self.alpha:
                incompatible_pairs.add(edge)
        return incompatible_pairs
    
    def generate_instance(self):
        W = self.generate_random_warehouse()
        self.generate_revenues(W)
        incompatible_pairs = self.generate_incompatibility(W)
        
        res = {'W': W, 'incompatible_pairs': incompatible_pairs}
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        W, incompatible_pairs = instance['W'], instance['incompatible_pairs']

        model = Model("SimpleWarehouseStockingProblem")
        item_vars = {f"x{item}": model.addVar(vtype="B", name=f"x{item}") for item in W.nodes}

        # Objective function: Maximize revenue
        objective_expr = quicksum(
            W.nodes[item]['revenue'] * item_vars[f"x{item}"]
            for item in W.nodes
        )

        # Incompatibility Constraints
        for u, v in incompatible_pairs:
            model.addCons(
                item_vars[f"x{u}"] + item_vars[f"x{v}"] <= 1,
                name=f"Incompat_{u}_{v}"
            )
        
        # Bin Capacity constraint
        model.addCons(
            quicksum(item_vars[f"x{item}"] for item in W.nodes) <= self.bin_capacity,
            name="BinCapacity"
        )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_items': 37,
        'max_items': 974,
        'incompat_prob': 0.47,
        'alpha': 0.74,
        'bin_capacity': 400,
    }
    
    swp = SimpleWarehouseStockingProblem(parameters, seed=seed)
    instance = swp.generate_instance()
    solve_status, solve_time = swp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")