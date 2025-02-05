import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NewCityPlanner:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_city_plan(self):
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        G = nx.erdos_renyi_graph(n=n_zones, p=self.er_prob, seed=self.seed)
        return G

    def generate_zone_utility_data(self, G):
        for node in G.nodes:
            G.nodes[node]['utility'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['distance'] = np.random.randint(1, 20)
    
    def generate_unwanted_routes(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E_invalid.add(edge)
        return E_invalid

    def find_zone_clusters(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_city_plan()
        self.generate_zone_utility_data(G)
        E_invalid = self.generate_unwanted_routes(G)
        clusters = self.find_zone_clusters(G)

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'clusters': clusters, 
        }
    
    def solve(self, instance):
        G, E_invalid, clusters = instance['G'], instance['E_invalid'], instance['clusters']
        
        model = Model("NewCityPlanner")
        zone_activation_vars = {f"z{node}":  model.addVar(vtype="B", name=f"z{node}") for node in G.nodes}
        cluster_route_usage_vars = {f"ru{u}_{v}": model.addVar(vtype="B", name=f"ru{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            G.nodes[node]['utility'] * zone_activation_vars[f"z{node}"]
            for node in G.nodes
        )
        
        objective_expr -= quicksum(
            G[u][v]['distance'] * cluster_route_usage_vars[f"ru{u}_{v}"]
            for u, v in E_invalid
        )

        model.setObjective(objective_expr, "maximize")

        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    zone_activation_vars[f"z{u}"] + zone_activation_vars[f"z{v}"] - cluster_route_usage_vars[f"ru{u}_{v}"] <= 1,
                    name=f"MallDistanceConstraints_{u}_{v}"
                )
            else:
                model.addCons(
                    zone_activation_vars[f"z{u}"] + zone_activation_vars[f"z{v}"] <= 1,
                    name=f"HousesClusterDistance_{u}_{v}"
                )

        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(zone_activation_vars[f"z{zone}"] for zone in cluster) <= 1,
                name=f"HouseDistance_{i}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 45,
        'max_zones': 700,
        'er_prob': 0.3,
        'beta': 0.8,
    }

    optimizer = NewCityPlanner(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")