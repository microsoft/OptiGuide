import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedGISP_NDP:
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
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_static_costs(self, G):
        for u, v in G.edges:
            G[u][v]['static_cost'] = G[u][v]['cost']
            G[u][v]['network_strength'] = np.random.uniform(0.5, 1.5)

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        self.generate_static_costs(G)
        
        # New energy consumption limits generation
        energy_limits = {node: np.random.randint(50, 150) for node in G.nodes}
        
        return {'G': G, 'E2': E2, 'energy_limits': energy_limits}
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, energy_limits = instance['G'], instance['E2'], instance['energy_limits']
        
        model = Model("Simplified_GISP_NDP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # New vars for energy consumption status
        energy_vars = {node: model.addVar(vtype="C", name=f"e{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['static_cost'] * edge_vars[f"y{u}_{v}"]
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
        
        # Energy consumption constraints
        for node in G.nodes:
            model.addCons(
                energy_vars[node] <= energy_limits[node],
                name=f"EnergyLimit_{node}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 37,
        'max_n': 292,
        'er_prob': 0.73,
        'set_type': 'SET1',
        'set_param': 1400.0,
        'alpha': 0.52,
    }

    simplified_gisp_ndp = SimplifiedGISP_NDP(parameters, seed=seed)
    instance = simplified_gisp_ndp.generate_instance()
    solve_status, solve_time = simplified_gisp_ndp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")