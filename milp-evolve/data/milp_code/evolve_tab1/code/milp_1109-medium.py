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
        G = None
        if self.graph_type == 'erdos_renyi':
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        elif self.graph_type == 'barabasi_albert':
            G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        elif self.graph_type == 'watts_strogatz':
            G = nx.watts_strogatz_graph(n=n_nodes, k=self.ws_k, p=self.ws_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.uniform(*self.revenue_range)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.uniform(*self.cost_range)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            u, v = edge
            node_degree_sum = G.degree[u] + G.degree[v]
            if np.random.random() <= (1 / (1 + np.exp(-self.beta * (node_degree_sum / 2 - self.degree_threshold)))):
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
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        z = model.addVar(vtype="B", name="z")

        # Objective: Maximize node revenues while minimizing edge costs and incorporating a subsidiary variable 'z'
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

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

        # Additional constraints to add complexity
        for node in G.nodes:
            model.addCons(
                quicksum(edge_vars[f"y{u}_{v}"] for u, v in G.edges if u == node or v == node) <= z,
                name=f"C_degree_limit_{node}"
            )

        model.setObjective(objective_expr + z * self.subsidiary_penalty, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 400,
        'max_n': 1300,
        'er_prob': 0.64,
        'graph_type': 'barabasi_albert',
        'ba_m': 5,
        'ws_k': 140,
        'ws_prob': 0.17,
        'revenue_range': (350, 1400),
        'cost_range': (3, 60),
        'beta': 52.5,
        'degree_threshold': 20,
        'subsidiary_penalty': 75.0,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")