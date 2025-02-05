import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GISP:
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
            G.nodes[node]['capacity'] = np.random.randint(1, self.max_capacity)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.normal(loc=self.norm_mean, scale=self.norm_sd)

    def generate_special_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_special_edges(G)
        res = {'G': G, 'E2': E2}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        
        model = Model("EnhancedGISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        extra_vars = {f"z{node}": model.addVar(vtype="I", lb=0, ub=G.nodes[node]['capacity'], name=f"z{node}") for node in G.nodes}
        special_edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in E2}

        # New complex objective with weighted sum of node revenues and edge costs
        objective_expr = quicksum(
            (G.nodes[node]['revenue'] * node_vars[f"x{node}"] + 0.5 * extra_vars[f"z{node}"])
            for node in G.nodes
        ) - quicksum(
            (G[u][v]['cost'] * special_edge_vars[f"y{u}_{v}"])
            for u, v in E2
        )

        # Capacity constraints for nodes
        for node in G.nodes:
            model.addCons(
                extra_vars[f"z{node}"] <= G.nodes[node]['capacity'] * node_vars[f"x{node}"],
                name=f"Capacity_{node}"
            )

        # Enhanced constraints considering node degrees and special edge interactions
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - special_edge_vars[f"y{u}_{v}"] + quicksum(node_vars[f"x{u}"] for u in G.neighbors(u)) <= 2,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 900,
        'max_n': 1040,
        'ba_m': 3,
        'gamma_shape': 18.0,
        'gamma_scale': 4.0,
        'norm_mean': 0.0,
        'norm_sd': 100.0,
        'max_capacity': 60,
        'alpha': 0.59,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")