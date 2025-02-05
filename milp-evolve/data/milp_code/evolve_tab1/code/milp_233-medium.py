import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EDO:  # Energy Distribution Optimization
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_buildings = np.random.randint(self.min_buildings, self.max_buildings)
        G = nx.erdos_renyi_graph(n=n_buildings, p=self.er_prob, seed=self.seed)
        return G

    def generate_buildings_energy_data(self, G):
        for node in G.nodes:
            G.nodes[node]['monthly_energy_demand'] = np.random.randint(50, 200)

        for u, v in G.edges:
            G[u][v]['transmission_cost'] = np.random.randint(5, 30)

    def generate_congested_pairs(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E_invalid.add(edge)
        return E_invalid

    def find_distribution_clusters(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_buildings_energy_data(G)
        E_invalid = self.generate_congested_pairs(G)
        clusters = self.find_distribution_clusters(G)
        return {
            'G': G,
            'E_invalid': E_invalid, 
            'clusters': clusters, 
        }
    
    def solve(self, instance):
        G, E_invalid, clusters = instance['G'], instance['E_invalid'], instance['clusters']
        
        model = Model("EDO")
        building_vars = {f"b{node}":  model.addVar(vtype="B", name=f"b{node}") for node in G.nodes}
        campaign_vars = {f"c{u}_{v}": model.addVar(vtype="B", name=f"c{u}_{v}") for u, v in G.edges}
        demand_met_vars = {f"d{node}": model.addVar(vtype="C", name=f"d{node}") for node in G.nodes}
        
        # New objective function: Maximize the energy supplied minus the transmission cost
        objective_expr = quicksum(
            G.nodes[node]['monthly_energy_demand'] * building_vars[f"b{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['transmission_cost'] * campaign_vars[f"c{u}_{v}"]
            for u, v in E_invalid
        )

        model.setObjective(objective_expr, "maximize")

        # Modify constraints
        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    building_vars[f"b{u}"] + building_vars[f"b{v}"] - campaign_vars[f"c{u}_{v}"] <= 1,
                    name=f"NodeCapacityConstraint_{u}_{v}"
                )
            else:
                model.addCons(
                    building_vars[f"b{u}"] + building_vars[f"b{v}"] <= 1,
                    name=f"NodeCapacityConstraint_{u}_{v}"
                )

        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(building_vars[f"b{building}"] for building in cluster) <= 1,
                name=f"Cluster_{i}"
            )

        for node in G.nodes:
            model.addCons(
                demand_met_vars[f"d{node}"] <= G.nodes[node]['monthly_energy_demand'] * building_vars[f"b{node}"],
                name=f"MonthlyDemandMet_{node}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_buildings': 65,
        'max_buildings': 616,
        'er_prob': 0.1,
        'beta': 0.17,
    }

    edo = EDO(parameters, seed=seed)
    instance = edo.generate_instance()
    solve_status, solve_time = edo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")