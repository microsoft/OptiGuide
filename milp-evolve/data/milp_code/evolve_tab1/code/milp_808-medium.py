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
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['cost'] = 1.0  # Simplified uniform cost

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
        res['node_capacities'] = {node: np.random.randint(self.min_capacity, self.max_capacity) for node in G.nodes}
        res['max_removable_edges'] = np.random.randint(self.min_removable_edges, self.max_removable_edges)
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        node_capacities = instance['node_capacities']
        max_removable_edges = instance['max_removable_edges']

        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # Objective: Maximize revenues and minimize costs
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        # Constraints
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

        # Set Partitioning constraints
        for node in G.nodes:
            model.addCons(
                quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2 if u == node or v == node) == node_vars[f"x{node}"],
                name=f"Partition_{node}"
            )

        # Knapsack constraints on nodes
        for node in G.nodes:
            model.addCons(
                quicksum(edge_vars[f"y{u}_{v}"] for u, v in G.edges if u == node or v == node) <= node_capacities[node],
                name=f"Knapsack_Node_{node}"
            )

        # Additional edge constraints
        model.addCons(
            quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2) <= max_removable_edges,
            name="Max_Removable_Edges"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 6,
        'max_n': 292,
        'set_type': 'SET1',
        'set_param': 3000.0,
        'alpha': 0.52,
        'ba_m': 15,
        'min_capacity': 1350,
        'max_capacity': 2355,
        'min_removable_edges': 123,
        'max_removable_edges': 2520,
    }
    
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")