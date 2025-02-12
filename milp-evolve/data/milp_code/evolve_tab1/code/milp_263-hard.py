import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GDO:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_households = np.random.randint(self.min_households, self.max_households)
        G = nx.erdos_renyi_graph(n=n_households, p=self.er_prob, seed=self.seed)
        return G

    def generate_households_market_data(self, G):
        for node in G.nodes:
            G.nodes[node]['nutrient_demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['transport_cost'] = np.random.randint(1, 20)

    def find_distribution_clusters(self, G): 
        cliques = list(nx.find_cliques(G))
        distribution_zones = [clique for clique in cliques if len(clique) > 1]
        return distribution_zones

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_households_market_data(G)
        clusters = self.find_distribution_clusters(G)

        return {
            'G': G,
            'clusters': clusters, 
        }
    
    def solve(self, instance):
        G, clusters = instance['G'], instance['clusters']
        
        model = Model("GDO")

        # Variables
        household_vars = {f"h{node}":  model.addVar(vtype="B", name=f"h{node}") for node in G.nodes}
        market_vars = {f"m{u}_{v}": model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}

        # Objective
        objective_expr = quicksum(G.nodes[node]['nutrient_demand'] * household_vars[f"h{node}"] for node in G.nodes)
        objective_expr -= quicksum(G[u][v]['transport_cost'] * market_vars[f"m{u}_{v}"] for u, v in G.edges)

        model.setObjective(objective_expr, "maximize")

        # Constraints
        for u, v in G.edges:
            model.addCons(
                household_vars[f"h{u}"] + household_vars[f"h{v}"] <= 1,
                name=f"Market_{u}_{v}"
            )

        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(household_vars[f"h{household}"] for household in cluster) <= 1,
                name=f"Cluster_{i}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_households': 130,
        'max_households': 205,
        'er_prob': 0.24,
    }

    gdo = GDO(parameters, seed=seed)
    instance = gdo.generate_instance()
    solve_status, solve_time = gdo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")