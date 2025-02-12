import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedFoodDistributionMILP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_distribution_graph(self):
        n_nodes = np.random.randint(self.min_stores, self.max_stores)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.connectivity_rate, directed=False, seed=self.seed)
        return G
    
    def generate_demand_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(100, 1000)

        for u, v in G.edges:
            G[u][v]['distance'] = np.random.randint(1, 100)
            G[u][v]['capacity'] = np.random.randint(10, 100)
    
    def define_coverage_zones(self, G):
        coverage_zones = list(nx.find_cliques(G))
        return coverage_zones
    
    def get_instance(self):
        G = self.generate_distribution_graph()
        self.generate_demand_data(G)
        coverage_zones = self.define_coverage_zones(G)

        warehouse_capacity = {node: np.random.randint(100, 1000) for node in G.nodes}
        
        return {
            'G': G,
            'coverage_zones': coverage_zones,
            'warehouse_capacity': warehouse_capacity
        }
    
    def solve(self, instance):
        G = instance['G']
        coverage_zones = instance['coverage_zones']
        warehouse_capacity = instance['warehouse_capacity']

        model = Model("SimplifiedFoodDistributionNetwork")
        
        # Define all variables
        distribution_vars = {f"DistCenter_{node}": model.addVar(vtype="B", name=f"DistCenter_{node}") for node in G.nodes}

        # Define objective
        objective_expr = quicksum(
            G.nodes[node]['demand'] * distribution_vars[f"DistCenter_{node}"]
            for node in G.nodes
        )

        # Applying Maximum Coverage Formulation
        for i, zone in enumerate(coverage_zones):
            for j in range(len(zone)):
                for k in range(j + 1, len(zone)):
                    u, v = zone[j], zone[k]
                    model.addCons(
                        distribution_vars[f"DistCenter_{u}"] + distribution_vars[f"DistCenter_{v}"] <= 1,
                        name=f"MaximumTransport_Zone_{i}_{u}_{v}"
                    )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_stores': 30,
        'max_stores': 800,
        'connectivity_rate': 0.42,
        'transport_budget': 10000,
        'no_of_scenarios': 1500,
    }
    
    distribution_optimization = SimplifiedFoodDistributionMILP(parameters, seed=seed)
    instance = distribution_optimization.get_instance()
    solve_status, solve_time = distribution_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")