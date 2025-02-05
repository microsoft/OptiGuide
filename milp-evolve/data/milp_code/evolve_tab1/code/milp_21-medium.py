"""
Generalized Independent Set Problem (GISP) with Diverse Constraints and Objective
"""
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
        graph_type = np.random.choice(['ER', 'BA'], p=[0.5, 0.5])
        if graph_type == 'ER':
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        else:
            G= nx.barabasi_albert_graph(n=n_nodes, m=self.ba_param, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.normal(loc=self.n_rev_mean, scale=self.n_rev_std)
        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(np.random.gamma(shape=self.edge_g_shape, scale=self.edge_g_scale))
            G[u][v]['weight'] = np.random.poisson(lam=self.edge_lambda)

    def generate_removable_edges(self, G):
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
        E2 = self.generate_removable_edges(G)
        cliques = self.find_maximal_cliques(G)
        res = {'G': G, 'E2': E2, 'cliques': cliques}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, cliques = instance['G'], instance['E2'], instance['cliques']
        
        model = Model("GISP_Diverse")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        
        # Objective function
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        ) - quicksum(
            G[u][v]['weight'] * edge_vars[f"y{u}_{v}"]
            for u, v in G.edges if (u, v) not in E2
        )
        
        # Edge constraints
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
        # Clique constraints
        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(node_vars[f"x{node}"] for node in clique) <= int(np.random.gamma(shape=self.clique_shape, scale=self.clique_scale)),
                name=f"Clique_{i}"
            )
        
        # Capacity constraints
        for node in G.nodes:
            model.addCons(
                node_vars[f"x{node}"] <= int(np.random.poisson(lam=self.node_capacity_lambda)),
                name=f"Capacity_{node}"
            )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 100,
        'max_n': 1300,
        'er_prob': 0.41,
        'ba_param': 20,
        'n_rev_mean': 50,
        'n_rev_std': 30,
        'edge_g_shape': 14,
        'edge_g_scale': 105,
        'edge_lambda': 80,
        'alpha': 0.8,
        'clique_shape': 60.0,
        'clique_scale': 75.0,
        'node_capacity_lambda': 500,
    }
    
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")