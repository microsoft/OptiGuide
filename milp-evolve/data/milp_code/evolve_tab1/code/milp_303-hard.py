import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedCityPlanner:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_city_plan(self):
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        G = nx.barabasi_albert_graph(n=n_zones, m=self.ba_m, seed=self.seed)
        return G

    def generate_zone_utility_data(self, G):
        for node in G.nodes:
            G.nodes[node]['utility'] = np.random.randint(1, 100)
            G.nodes[node]['capacity'] = np.random.randint(1, 10)
            G.nodes[node]['type'] = np.random.choice(['residential', 'commercial', 'industrial'])

        for u, v in G.edges:
            G[u][v]['distance'] = np.random.randint(1, 20)
            G[u][v]['traffic'] = np.random.poisson(lam=5)
    
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
        
        model = Model("EnhancedCityPlanner")
        zone_activation_vars = {f"z{node}":  model.addVar(vtype="B", name=f"z{node}") for node in G.nodes}
        route_usage_vars = {f"ru{u}_{v}": model.addVar(vtype="B", name=f"ru{u}_{v}") for u, v in G.edges}
        zone_capacity_vars = {f"c{node}": model.addVar(vtype="C", lb=0, ub=G.nodes[node]['capacity'], name=f"c{node}") for node in G.nodes}

        # Enhanced objective function
        objective_expr = quicksum(
            G.nodes[node]['utility'] * zone_activation_vars[f"z{node}"]
            for node in G.nodes
        )
        
        objective_expr -= quicksum(
            G[u][v]['distance'] * route_usage_vars[f"ru{u}_{v}"] + G[u][v]['traffic']
            for u, v in E_invalid
        )

        # Added capacity utilization element to the objective
        objective_expr += quicksum(
            zone_capacity_vars[f"c{node}"] * 5
            for node in G.nodes
        )

        model.setObjective(objective_expr, "maximize")

        # Enhanced constraints
        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    zone_activation_vars[f"z{u}"] + zone_activation_vars[f"z{v}"] - route_usage_vars[f"ru{u}_{v}"] <= 1,
                    name=f"InvalidDistanceConstraints_{u}_{v}"
                )

        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(zone_activation_vars[f"z{zone}"] for zone in cluster) <= 1,
                name=f"ClusterConstraint_{i}"
            )

        # Capacity constraints
        for node in G.nodes:
            model.addCons(
                zone_capacity_vars[f"c{node}"] <= zone_activation_vars[f"z{node}"] * G.nodes[node]['capacity'],
                name=f"CapacityConstraint_{node}"
            )

        # Constraints to enforce commercial zones with additional benefits
        for node in G.nodes:
            if G.nodes[node]['type'] == 'commercial':
                model.addCons(
                    zone_activation_vars[f"z{node}"] * 3 <= zone_capacity_vars[f"c{node}"],
                    name=f"CommercialZoneBenefit_{node}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 225,
        'max_zones': 500,
        'er_prob': 0.6,
        'beta': 0.8,
        'ba_m': 15,
    }

    optimizer = EnhancedCityPlanner(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")