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
        if self.set_type == 'SET1':
            for node in G.nodes:
                G.nodes[node]['revenue'] = np.random.randint(1, 100)
            for u, v in G.edges:
                G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)
        elif self.set_type == 'SET2':
            for node in G.nodes:
                G.nodes[node]['revenue'] = float(self.set_param)
            for u, v in G.edges:
                G[u][v]['cost'] = 1.0

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
        res['weather_impact'] = {(u, v): np.random.uniform(0, 1) for u, v in G.edges}
        res['hazard_levels'] = {(u, v): np.random.randint(0, 2) for u, v in G.edges}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        node_capacities = instance['node_capacities']
        max_removable_edges = instance['max_removable_edges']
        weather_impact = instance['weather_impact']
        hazard_levels = instance['hazard_levels']

        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        weather_vars = {f"w{u}_{v}": model.addVar(vtype="B", name=f"w{u}_{v}") for u, v in G.edges}
        hazard_vars = {f"h{u}_{v}": model.addVar(vtype="B", name=f"h{u}_{v}") for u, v in G.edges}

        # Objective: Maximize revenues and minimize costs
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        # Original constraints
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

        # New Constraints using Set Partitioning
        for node in G.nodes:
            model.addCons(
                quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2 if u == node or v == node) == node_vars[f"x{node}"],
                name=f"Partition_{node}"
            )
        
        # New knapsack constraints on nodes
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

        # Weather impact constraints
        for u, v in G.edges:
            model.addCons(
                edge_vars[f"y{u}_{v}"] <= weather_vars[f"w{u}_{v}"] + (1 - weather_impact[u, v]),
                name=f"Weather_Impact_{u}_{v}"
            )

        # Hazard avoidance constraints
        for u, v in G.edges:
            model.addCons(
                edge_vars[f"y{u}_{v}"] <= hazard_vars[f"h{u}_{v}"] + (1 - hazard_levels[u, v]),
                name=f"Hazard_Avoidance_{u}_{v}"
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
        'max_n': 780,
        'er_prob': 0.59,
        'set_type': 'SET1',
        'set_param': 3000.0,
        'alpha': 0.8,
        'ba_m': 7,
        'min_capacity': 60,
        'max_capacity': 70,
        'min_removable_edges': 45,
        'max_removable_edges': 300,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")