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

    def generate_healthcare_providers(self, G):
        healthcare_providers = {}
        for node in G.nodes:
            healthcare_providers[node] = np.random.randint(1, self.n_healthcare_providers)
        return healthcare_providers

    def generate_temperatures(self, G):
        temperatures = {}
        for node in G.nodes:
            temperatures[node] = np.random.uniform(self.min_temp, self.max_temp)
        return temperatures

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        healthcare_providers = self.generate_healthcare_providers(G)
        temperatures = self.generate_temperatures(G)
        res = {'G': G, 'E2': E2, 'healthcare_providers': healthcare_providers, 'temperatures': temperatures}
        
        for u, v in G.edges:
            break_points = np.linspace(0, G[u][v]['cost'], self.num_pieces+1)
            slopes = np.diff(break_points)
            res[f'break_points_{u}_{v}'] = break_points
            res[f'slopes_{u}_{v}'] = slopes
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        healthcare_providers = instance['healthcare_providers']
        temperatures = instance['temperatures']
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piece_vars = {f"t{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"t{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_pieces)}
        healthcare_vars = {f"hp_{node}": model.addVar(vtype="B", name=f"hp_{node}") for node in G.nodes}
        temperature_vars = {f"temp_{node}": model.addVar(vtype="C", lb=self.min_temp, ub=self.max_temp, name=f"temp_{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            instance[f'slopes_{u}_{v}'][k] * piece_vars[f"t{u}_{v}_{k}"]
            for u, v in E2
            for k in range(self.num_pieces)
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
                
        for u, v in E2:
            model.addCons(quicksum(piece_vars[f"t{u}_{v}_{k}"] for k in range(self.num_pieces)) == edge_vars[f'y{u}_{v}'])
            
            for k in range(self.num_pieces):
                model.addCons(piece_vars[f"t{u}_{v}_{k}"] <= instance[f'break_points_{u}_{v}'][k+1] - instance[f'break_points_{u}_{v}'][k])
        
        # Healthcare Provider assignment constraints
        for node in G.nodes:
            model.addCons(healthcare_vars[f"hp_{node}"] <= node_vars[f"x{node}"], name=f"HPA_{node}")
        
        # Temperature control constraints
        for node in G.nodes:
            model.addCons(temperature_vars[f"temp_{node}"] >= temperatures[node] * node_vars[f"x{node}"], name=f"Temp_{node}")
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 25,
        'max_n': 455,
        'er_prob': 0.38,
        'set_type': 'SET1',
        'set_param': 3000.0,
        'alpha': 0.73,
        'num_pieces': 18,
        'n_healthcare_providers': 100,
        'min_temp': 6,
        'max_temp': 16,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")