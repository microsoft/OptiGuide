import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WDO:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_neighborhoods = np.random.randint(self.min_neighborhoods, self.max_neighborhoods)
        G = nx.erdos_renyi_graph(n=n_neighborhoods, p=self.er_prob, seed=self.seed)
        return G

    def generate_neighborhood_data(self, G):
        for node in G.nodes:
            G.nodes[node]['water_demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['segments'] = [((i + 1) * 10, np.random.randint(1, 10)) for i in range(self.num_segments)]

    def find_distribution_zones(self, G): 
        cliques = list(nx.find_cliques(G))
        distribution_zones = [clique for clique in cliques if len(clique) > 1]
        return distribution_zones

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_neighborhood_data(G)
        zones = self.find_distribution_zones(G)

        return {
            'G': G,
            'zones': zones, 
        }
    
    def solve(self, instance):
        G, zones = instance['G'], instance['zones']
        
        model = Model("WDO")

        # Variables
        neighborhood_vars = {f"n{node}": model.addVar(vtype="B", name=f"n{node}") for node in G.nodes}
        manufacturer_vars = {(u, v): model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}

        zonal_transport_vars = {}
        for u, v in G.edges:
            for i in range(self.num_segments):
                zonal_transport_vars[(u, v, i)] = model.addVar(vtype="C", name=f"zonal_transport_{u}_{v}_{i}")

        # Objective
        objective_expr = quicksum(G.nodes[node]['water_demand'] * neighborhood_vars[f"n{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr -= zonal_transport_vars[(u, v, i)] * cost

        model.setObjective(objective_expr, "maximize")

        # Constraints
        for u, v in G.edges:
            model.addCons(
                neighborhood_vars[f"n{u}"] + neighborhood_vars[f"n{v}"] <= 1,
                name=f"Manufacturer_{u}_{v}"
            )
            model.addCons(
                quicksum(zonal_transport_vars[(u, v, i)] for i in range(self.num_segments)) == manufacturer_vars[(u, v)] * 100,
                name=f"ZonalTransport_{u}_{v}"
            )

        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(neighborhood_vars[f"n{neighborhood}"] for neighborhood in zone) <= 1,
                name=f"ZonalSupply_{i}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_neighborhoods': 50,
        'max_neighborhoods': 300,
        'er_prob': 0.31,
        'num_segments': 3,
    }

    wdo = WDO(parameters, seed=seed)
    instance = wdo.generate_instance()
    solve_status, solve_time = wdo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")