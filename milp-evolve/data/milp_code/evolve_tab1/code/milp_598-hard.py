import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FoodDistributionMILP:
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
            G.nodes[node]['demand'] = np.random.randint(50, 1000)

        for u, v in G.edges:
            G[u][v]['distance'] = np.random.randint(1, 100)
            G[u][v]['capacity'] = np.random.randint(10, 100)
    
    def generate_hazard_zones(self, G):
        H_spots = set()
        for node in G.nodes:
            if np.random.random() <= self.hazard_rate:
                H_spots.add(node)
        return H_spots
    
    def define_coverage_zones(self, G):
        coverage_zones = list(nx.find_cliques(G))
        return coverage_zones
    
    def get_instance(self):
        G = self.generate_distribution_graph()
        self.generate_demand_data(G)
        H_spots = self.generate_hazard_zones(G)
        coverage_zones = self.define_coverage_zones(G)

        warehouse_capacity = {node: np.random.randint(50, 500) for node in G.nodes}
        transport_cost = {(u, v): np.random.uniform(5.0, 20.0) for u, v in G.edges}
        handling_cost = {node: np.random.uniform(10.0, 150.0) for node in G.nodes}
        
        return {
            'G': G,
            'H_spots': H_spots, 
            'coverage_zones': coverage_zones, 
            'warehouse_capacity': warehouse_capacity, 
            'transport_cost': transport_cost, 
            'handling_cost': handling_cost
        }
    
    def solve(self, instance):
        G, H_spots, coverage_zones = instance['G'], instance['H_spots'], instance['coverage_zones']
        warehouse_capacity = instance['warehouse_capacity']
        transport_cost = instance['transport_cost']
        handling_cost = instance['handling_cost']

        model = Model("FoodDistributionNetwork")
        
        # Define all variables
        distribution_vars = {f"DistCenter_{node}": model.addVar(vtype="B", name=f"DistCenter_{node}") for node in G.nodes}
        transport_vars = {f"Transport_{u}_{v}": model.addVar(vtype="B", name=f"Transport_{u}_{v}") for u, v in G.edges}
        transport_budget = model.addVar(vtype="C", name="TransportBudget")

        # Define objective
        objective_expr = quicksum(
            G.nodes[node]['demand'] * distribution_vars[f"DistCenter_{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['distance'] * transport_vars[f"Transport_{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            transport_cost[(u, v)] * transport_vars[f"Transport_{u}_{v}"]
            for u, v in G.edges
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

        M = 1000  # Big M constant, set contextually larger than any decision boundary.

        for u, v in G.edges:
            # Connectivity Rate Constraints replacing Big-M
            model.addCons(
                distribution_vars[f"DistCenter_{u}"] + distribution_vars[f"DistCenter_{v}"] <= 1 + transport_vars[f"Transport_{u}_{v}"],
                name=f"MaximumTransport_{u}_{v}"
            )
            model.addCons(
                distribution_vars[f"DistCenter_{u}"] + distribution_vars[f"DistCenter_{v}"] >= 2 * transport_vars[f"Transport_{u}_{v}"] - transport_vars[f"Transport_{u}_{v}"],
                name=f"MaximumTransport_{u}_{v}_other"
            )

        # Hazard constraints
        for node in H_spots:
            model.addCons(
                distribution_vars[f"DistCenter_{node}"] <= warehouse_capacity[node],
                name=f"HazardSpots_{node}"
            )

        model.addCons(
            transport_budget <= self.transport_budget,
            name="CostCap_Limit"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_stores': 40,
        'max_stores': 900,
        'connectivity_rate': 0.2,
        'hazard_rate': 0.1,
        'transport_budget': 5000,
        'no_of_scenarios': 350,
    }
    
    distribution_optimization = FoodDistributionMILP(parameters, seed=seed)
    instance = distribution_optimization.get_instance()
    solve_status, solve_time = distribution_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")