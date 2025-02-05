import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedGISP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.normal(loc=self.norm_mean, scale=self.norm_sd)

    def generate_special_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def find_maximal_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        cliques = self.find_maximal_cliques(G)
        
        res = {'G': G, 'cliques': cliques}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, cliques = instance['G'], instance['cliques']

        model = Model("SimplifiedGISP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # Simplified objective
        objective_expr = quicksum(G.nodes[node]['revenue'] * node_vars[f"x{node}"] for node in G.nodes) - quicksum(G[u][v]['cost'] * edge_vars[f"y{u}_{v}"] for u, v in G.edges)

        # Existing node-edge constraints
        for u, v in G.edges:
            model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1, name=f"C_{u}_{v}")

        # Adding clique constraints
        for i, clique in enumerate(cliques):
            model.addCons(quicksum(node_vars[f"x{node}"] for node in clique) <= 1, name=f"Clique_{i}")

        # Objective and solve
        model.setObjective(objective_expr, "maximize")
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 900,
        'max_n': 1500,
        'ba_m': 8,
        'gamma_shape': 0.74,
        'gamma_scale': 252.0,
        'norm_mean': 0.0,
        'norm_sd': 2500.0,
        'alpha': 0.26,
    }
    
    gisp = SimplifiedGISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")