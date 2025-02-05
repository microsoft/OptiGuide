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
            G.nodes[node]['revenue'] = np.random.uniform(10, 200)
            G.nodes[node]['resource_limit'] = np.random.uniform(50, 150)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.normal(loc=10.0, scale=5.0)
            G[u][v]['connection_strength'] = np.random.uniform(1.0, 10.0)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_logical_conditions(self, G):
        logical_conditions = {}
        node_pairs = list(nx.non_edges(G))
        np.random.shuffle(node_pairs)
        for i in range(int(self.logical_condition_ratio * len(G.nodes))):
            u, v = node_pairs[i]
            logical_conditions[(u, v)] = np.random.choice(["AND", "OR"])
        return logical_conditions

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        logical_conditions = self.generate_logical_conditions(G)
        res = {'G': G, 'E2': E2, 'logical_conditions': logical_conditions}
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, logical_conditions = instance['G'], instance['E2'], instance['logical_conditions']
        
        model = Model("Enhanced_GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        resource_vars = {f"r{node}": model.addVar(lb=0, ub=G.nodes[node]['resource_limit'], name=f"r{node}") for node in G.nodes}
        strength_vars = {f"s{u}_{v}": model.addVar(lb=0, ub=G[u][v]['connection_strength'], name=f"s{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        ) + quicksum(
            G[u][v]['connection_strength'] * strength_vars[f"s{u}_{v}"]
            for u, v in G.edges
        ) - quicksum(
            resource_vars[f"r{node}"]
            for node in G.nodes
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

        # Additional Constraints
        for node in G.nodes:
            model.addCons(
                resource_vars[f"r{node}"] >= node_vars[f"x{node}"] * G.nodes[node]['resource_limit'],
                name=f"Resource_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                strength_vars[f"s{u}_{v}"] <= node_vars[f"x{u}"] + node_vars[f"x{v}"],
                name=f"Strength_{u}_{v}"
            )

        for (u, v), condition in logical_conditions.items():
            if condition == "AND":
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"Logical_AND_{u}_{v}"
                )
            elif condition == "OR":
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] >= 1,
                    name=f"Logical_OR_{u}_{v}"
                )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 500,
        'max_n': 1950,
        'ba_m': 5,
        'alpha': 0.66,
        'logical_condition_ratio': 0.24,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")