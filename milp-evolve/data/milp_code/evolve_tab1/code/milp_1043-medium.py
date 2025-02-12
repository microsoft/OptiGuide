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
            G.nodes[node]['revenue'] = np.random.gamma(shape=self.revenue_gamma_shape, scale=self.revenue_gamma_scale)
        for u, v in G.edges:
            G[u][v]['cost'] = abs(G.nodes[u]['revenue'] - G.nodes[v]['revenue']) + np.random.normal(loc=self.edge_cost_mean, scale=self.edge_cost_sd)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        res = {'G': G, 'E2': E2}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        
        model = Model("AdvancedGISP")

        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        cost_vars = {f"c{u}_{v}": model.addVar(lb=0.0, ub=G[u][v]['cost'] * 2, vtype="C", name=f"c{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            cost_vars[f"c{u}_{v}"]
            for u, v in G.edges
        )

        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
                model.addCons(
                    cost_vars[f"c{u}_{v}"] >= edge_vars[f"y{u}_{v}"] * G[u][v]['cost'],
                    name=f"D_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"E_{u}_{v}"
                )
                model.addCons(
                    cost_vars[f"c{u}_{v}"] >= 0,
                    name=f"F_{u}_{v}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 300,
        'max_n': 1300,
        'ba_m': 100,
        'revenue_gamma_shape': 18.0,
        'revenue_gamma_scale': 250.0,
        'edge_cost_mean': 90,
        'edge_cost_sd': 35,
        'alpha': 0.56,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")