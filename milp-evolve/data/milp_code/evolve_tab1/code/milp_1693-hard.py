import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class VaccineDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_production_zones > 0 and self.n_distribution_zones > 0
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_production_capacity > 0 and self.max_production_capacity >= self.min_production_capacity

        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, self.n_production_zones)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_production_zones, self.n_distribution_zones))
        production_capacities = np.random.randint(self.min_production_capacity, self.max_production_capacity + 1, self.n_production_zones)
        distribution_demands = np.random.randint(1, 10, self.n_distribution_zones)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.n_production_zones)
        distances = np.random.uniform(0, self.max_transport_distance, (self.n_production_zones, self.n_distribution_zones))
        shipment_costs = np.random.randint(1, 20, (self.n_production_zones, self.n_distribution_zones))
        shipment_capacities = np.random.randint(5, 15, (self.n_production_zones, self.n_distribution_zones))

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.n_production_zones):
            for d in range(self.n_distribution_zones):
                G.add_edge(f"production_{p}", f"distribution_{d}")
                node_pairs.append((f"production_{p}", f"distribution_{d}"))

        return {
            "zone_costs": zone_costs,
            "transport_costs": transport_costs,
            "production_capacities": production_capacities,
            "distribution_demands": distribution_demands,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "shipment_costs": shipment_costs,
            "shipment_capacities": shipment_capacities,
        }

    def solve(self, instance):
        zone_costs = instance['zone_costs']
        transport_costs = instance['transport_costs']
        production_capacities = instance['production_capacities']
        distribution_demands = instance['distribution_demands']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        shipment_costs = instance['shipment_costs']
        shipment_capacities = instance['shipment_capacities']

        model = Model("VaccineDistributionOptimization")
        n_production_zones = len(zone_costs)
        n_distribution_zones = len(transport_costs[0])

        # Decision variables
        vaccine_vars = {p: model.addVar(vtype="B", name=f"Production_{p}") for p in range(n_production_zones)}
        flow_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total cost including production zone costs, transport costs, and shipment costs.
        model.setObjective(
            quicksum(zone_costs[p] * vaccine_vars[p] for p in range(n_production_zones)) +
            quicksum(transport_costs[p, int(v.split('_')[1])] * flow_vars[(u, v)] for (u, v) in node_pairs for p in range(n_production_zones) if u == f'production_{p}') +
            quicksum(shipment_costs[int(u.split('_')[1]), int(v.split('_')[1])] * flow_vars[(u, v)] for (u, v) in node_pairs),
            "minimize"
        )

        # Flow conservation constraints
        for p in range(n_production_zones):
            model.addCons(
                quicksum(flow_vars[(f"production_{p}", f"distribution_{d}")] for d in range(n_distribution_zones)) <= production_capacities[p] * vaccine_vars[p],
                f"Production_{p}_FlowCapacity"
            )
            model.addCons(
                quicksum(flow_vars[(f"production_{p}", f"distribution_{d}")] for d in range(n_distribution_zones)) <= budget_limits[p],
                f"Production_{p}_BudgetLimit"
            )

        for d in range(n_distribution_zones):
            model.addCons(
                quicksum(flow_vars[(f"production_{p}", f"distribution_{d}")] for p in range(n_production_zones)) == distribution_demands[d],
                f"Distribution_{d}_FlowBalance"
            )

        # Constraints: Shipment cannot exceed its capacity
        for u, v in node_pairs:
            model.addCons(flow_vars[(u, v)] <= shipment_capacities[int(u.split('_')[1]), int(v.split('_')[1])], f"FlowCapacity_{u}_{v}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_production_zones': 196,
        'n_distribution_zones': 56,
        'min_transport_cost': 480,
        'max_transport_cost': 2025,
        'min_zone_cost': 603,
        'max_zone_cost': 1050,
        'min_production_capacity': 2,
        'max_production_capacity': 2800,
        'min_budget_limit': 16,
        'max_budget_limit': 120,
        'max_transport_distance': 528,
    }
    
    optimizer = VaccineDistributionOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")