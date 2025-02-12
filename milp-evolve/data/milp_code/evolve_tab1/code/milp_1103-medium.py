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
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
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

    def generate_sensor_data(self, G):
        sensor_statuses = {node: np.random.choice([0, 1]) for node in G.nodes}
        return sensor_statuses

    def generate_maintenance_costs(self, G):
        maintenance_costs = {node: np.random.randint(10, 50) for node in G.nodes}
        return maintenance_costs

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        sensor_statuses = self.generate_sensor_data(G)
        maintenance_costs = self.generate_maintenance_costs(G)
        
        return {'G': G, 'E2': E2, 'sensor_statuses': sensor_statuses, 'maintenance_costs': maintenance_costs}
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, sensor_statuses, maintenance_costs = instance['G'], instance['E2'], instance['sensor_statuses'], instance['maintenance_costs']
        
        model = Model("GISP")
        
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        maintenance_vars = {f"m{node}": model.addVar(vtype="B", name=f"m{node}") for node in G.nodes}
        sensor_vars = {f"s{node}": model.addVar(vtype="B", name=f"s{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )
        
        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        objective_expr -= quicksum(
            maintenance_costs[node] * maintenance_vars[f"m{node}"]
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

        for node in G.nodes:
            model.addCons(
                node_vars[f"x{node}"] + sensor_vars[f"s{node}"] <= 1,
                name=f"S_{node}"
            )
            model.addCons(
                maintenance_vars[f"m{node}"] >= sensor_vars[f"s{node}"],
                name=f"M_{node}"
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
        'max_n': 130,
        'er_prob': 0.78,
        'set_type': 'SET1',
        'set_param': 800.0,
        'alpha': 0.59,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")