'''
Simplified Generalized Independent Set Problem (GISP) without Removable Edges
'''
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
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)

    def find_maximal_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues(G)
        cliques = self.find_maximal_cliques(G)
        res = {'G': G, 'cliques': cliques}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, cliques = instance['G'], instance['cliques']
        
        model = Model("SimplifiedGISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                name=f"C_{u}_{v}"
            )

        # Adding clique constraints
        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(node_vars[f"x{node}"] for node in clique) <= 1,
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
        'max_n': 325,
        'er_prob': 0.31,
    }
    
    simplified_gisp = SimplifiedGISP(parameters, seed=seed)
    instance = simplified_gisp.generate_instance()
    solve_status, solve_time = simplified_gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")