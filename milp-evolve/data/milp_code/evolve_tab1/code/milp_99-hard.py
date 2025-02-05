import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WarehouseStockingProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_random_warehouse(self):
        n_items = np.random.randint(self.min_n, self.max_n)
        W = nx.erdos_renyi_graph(n=n_items, p=self.compat_prob, seed=self.seed)
        return W

    def generate_revenues_costs(self, W):
        for item in W.nodes:
            W.nodes[item]['revenue'] = np.random.randint(1, 100)
        for u, v in W.edges:
            W[u][v]['compatibility'] = (W.nodes[u]['revenue'] + W.nodes[v]['revenue']) / float(self.compat_param)

    def generate_compatibility(self, W):
        incompatible_pairs = set()
        for edge in W.edges:
            if np.random.random() <= self.alpha:
                incompatible_pairs.add(edge)
        return incompatible_pairs

    def find_cliques(self, W):
        cliques = list(nx.find_cliques(W))
        return cliques

    def generate_instance(self):
        W = self.generate_random_warehouse()
        self.generate_revenues_costs(W)
        incompatible_pairs = self.generate_compatibility(W)
        cliques = self.find_cliques(W)
        
        res = {'W': W, 'incompatible_pairs': incompatible_pairs, 'cliques': cliques}
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        W, incompatible_pairs, cliques = instance['W'], instance['incompatible_pairs'], instance['cliques']

        model = Model("WarehouseStockingProblem")
        item_vars = {f"x{item}": model.addVar(vtype="B", name=f"x{item}") for item in W.nodes}
        compat_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in W.edges}

        # Objective function: Maximize revenue, minimize incompatibility
        objective_expr = quicksum(
            W.nodes[item]['revenue'] * item_vars[f"x{item}"]
            for item in W.nodes
        )

        objective_expr -= quicksum(
            W[u][v]['compatibility'] * compat_vars[f"y{u}_{v}"]
            for u, v in incompatible_pairs
        )

        # Logical compatibility constraints
        for u, v in W.edges:
            if (u, v) in incompatible_pairs:
                model.addCons(
                    item_vars[f"x{u}"] + item_vars[f"x{v}"] - compat_vars[f"y{u}_{v}"] <= 1,
                    name=f"Compat_{u}_{v}"
                )
            else:
                model.addCons(
                    item_vars[f"x{u}"] + item_vars[f"x{v}"] <= 1,
                    name=f"Compat_{u}_{v}"
                )
        
        # Bin Capacity constraint
        model.addCons(
            quicksum(item_vars[f"x{item}"] for item in W.nodes) <= self.bin_capacity,
            name="BinCapacity"
        )

        # Adding clique constraints
        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(item_vars[f"x{item}"] for item in clique) <= 1,
                name=f"Clique_{i}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 37,
        'max_n': 487,
        'compat_prob': 0.38,
        'compat_param': 562.5,
        'alpha': 0.66,
        'bin_capacity': 1000,
    }
    
    wsp = WarehouseStockingProblem(parameters, seed=seed)
    instance = wsp.generate_instance()
    solve_status, solve_time = wsp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")