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

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        res = {'G': G, 'E2': E2}
        
        # Parking zone data
        parking_capacity = np.random.randint(1, self.max_parking_capacity, size=self.n_parking_zones)
        parking_zones = {i: np.random.choice(range(len(G.nodes)), size=self.n_parking_in_zone, replace=False) for i in
                         range(self.n_parking_zones)}
        res.update({'parking_capacity': parking_capacity, 'parking_zones': parking_zones})

        # Delivery time windows with uncertainty
        time_windows = {node: (np.random.randint(0, self.latest_delivery_time // 2), 
                               np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for node in G.nodes}
        uncertainty = {node: np.random.normal(0, self.time_uncertainty_stddev, size=2) for node in G.nodes}
        res.update({'time_windows': time_windows, 'uncertainty': uncertainty})

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        parking_capacity, parking_zones = instance['parking_capacity'], instance['parking_zones']
        time_windows, uncertainty = instance['time_windows'], instance['uncertainty']
        
        model = Model("GISP")

        # Variables
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        parking_vars = {f"p{node}": model.addVar(vtype="B", name=f"p{node}") for node in G.nodes}
        time_vars = {f"t{node}": model.addVar(vtype="C", name=f"t{node}") for node in G.nodes}
        early_penalty_vars = {f"e{node}": model.addVar(vtype="C", name=f"e{node}") for node in G.nodes}
        late_penalty_vars = {f"l{node}": model.addVar(vtype="C", name=f"l{node}") for node in G.nodes}
        
        # Objective
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )
        objective_expr -= self.time_penalty_weight * quicksum(
            early_penalty_vars[f"e{node}"] + late_penalty_vars[f"l{node}"]
            for node in G.nodes
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

        # Parking constraints
        for zone, nodes in parking_zones.items():
            model.addCons(
                quicksum(parking_vars[f"p{node}"] for node in nodes if f"p{node}" in parking_vars) <= parking_capacity[zone], 
                f"parking_limit_{zone}"
            )
            for node in nodes:
                if f"p{node}" in parking_vars:
                    model.addCons(
                        parking_vars[f"p{node}"] <= node_vars[f"x{node}"], 
                        f"assign_parking_{node}"
                    )

        # Time windows
        for node in G.nodes:
            if f"t{node}" in time_vars:
                start_window, end_window = time_windows[node]
                uncertainty_start, uncertainty_end = uncertainty[node]
                model.addCons(time_vars[f"t{node}"] >= start_window + uncertainty_start, 
                              f"time_window_start_{node}")
                model.addCons(time_vars[f"t{node}"] <= end_window + uncertainty_end, 
                              f"time_window_end_{node}")
                
                model.addCons(early_penalty_vars[f"e{node}"] >= start_window + uncertainty_start - time_vars[f"t{node}"], 
                              f"early_penalty_{node}")
                model.addCons(late_penalty_vars[f"l{node}"] >= time_vars[f"t{node}"] - (end_window + uncertainty_end), 
                              f"late_penalty_{node}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 50,
        'max_n': 910,
        'er_prob': 0.31,
        'set_type': 'SET1',
        'set_param': 75.0,
        'alpha': 0.17,
        'max_parking_capacity': 2000,
        'n_parking_zones': 50,
        'n_parking_in_zone': 75,
        'latest_delivery_time': 720,
        'time_uncertainty_stddev': 7,
        'time_penalty_weight': 0.1,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")