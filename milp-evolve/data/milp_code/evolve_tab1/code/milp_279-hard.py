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
            # Generating piecewise linear segments for transport costs
            G[u][v]['segments'] = [((i + 1) * 10, np.random.randint(1, 10)) for i in range(self.num_segments)]

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
        household_vars = {f"h{node}": model.addVar(vtype="B", name=f"h{node}") for node in G.nodes}
        market_vars = {(u, v): model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}

        transport_vars = {}
        for u, v in G.edges:
            for i in range(self.num_segments):
                transport_vars[(u, v, i)] = model.addVar(vtype="C", name=f"transport_{u}_{v}_{i}")

        # Objective
        objective_expr = quicksum(G.nodes[node]['nutrient_demand'] * household_vars[f"h{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr -= transport_vars[(u, v, i)] * cost

        model.setObjective(objective_expr, "maximize")

        # Constraints
        for u, v in G.edges:
            model.addCons(
                household_vars[f"h{u}"] + household_vars[f"h{v}"] <= 1,
                name=f"Market_{u}_{v}"
            )
            model.addCons(
                quicksum(transport_vars[(u, v, i)] for i in range(self.num_segments)) == market_vars[(u, v)] * 100,
                name=f"Transport_{u}_{v}"
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
        'num_segments': 3,
    }

    gdo = GDO(parameters, seed=seed)
    instance = gdo.generate_instance()
    solve_status, solve_time = gdo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")