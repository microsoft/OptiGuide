import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DisasterReliefLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_supply_zones > 0 and self.n_relief_zones > 0
        assert self.min_supply_cost >= 0 and self.max_supply_cost >= self.min_supply_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_supply_capacity > 0 and self.max_supply_capacity >= self.min_supply_capacity

        supply_costs = np.random.randint(self.min_supply_cost, self.max_supply_cost + 1, self.n_supply_zones)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_supply_zones, self.n_relief_zones))
        supply_capacities = np.random.randint(self.min_supply_capacity, self.max_supply_capacity + 1, self.n_supply_zones)
        relief_demands = np.random.randint(1, 10, self.n_relief_zones)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.n_supply_zones)
        distances = np.random.uniform(0, self.max_transport_distance, (self.n_supply_zones, self.n_relief_zones))

        G = nx.DiGraph()
        node_pairs = []
        for s in range(self.n_supply_zones):
            for r in range(self.n_relief_zones):
                G.add_edge(f"supply_{s}", f"relief_{r}")
                node_pairs.append((f"supply_{s}", f"relief_{r}"))
                
        return {
            "supply_costs": supply_costs,
            "transport_costs": transport_costs,
            "supply_capacities": supply_capacities,
            "relief_demands": relief_demands,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        supply_costs = instance['supply_costs']
        transport_costs = instance['transport_costs']
        supply_capacities = instance['supply_capacities']
        relief_demands = instance['relief_demands']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("DisasterReliefLogisticsOptimization")
        n_supply_zones = len(supply_costs)
        n_relief_zones = len(transport_costs[0])
        
        # Decision variables
        supply_vars = {s: model.addVar(vtype="B", name=f"Supply_{s}") for s in range(n_supply_zones)}
        transport_vars = {(u, v): model.addVar(vtype="C", name=f"Transport_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total cost including supply zone costs and transport costs.
        model.setObjective(
            quicksum(supply_costs[s] * supply_vars[s] for s in range(n_supply_zones)) +
            quicksum(transport_costs[s, int(v.split('_')[1])] * transport_vars[(u, v)] for (u, v) in node_pairs for s in range(n_supply_zones) if u == f'supply_{s}'),
            "minimize"
        )

        # Relief distribution constraint for each zone
        for r in range(n_relief_zones):
            model.addCons(
                quicksum(transport_vars[(u, f"relief_{r}")] for u in G.predecessors(f"relief_{r}")) == relief_demands[r], 
                f"Relief_{r}_NodeFlowConservation"
            )

        # Constraints: Zones only receive supplies if the supply zones are operational
        for s in range(n_supply_zones):
            for r in range(n_relief_zones):
                model.addCons(
                    transport_vars[(f"supply_{s}", f"relief_{r}")] <= budget_limits[s] * supply_vars[s], 
                    f"Supply_{s}_SupplyLimitByBudget_{r}"
                )

        # Constraints: Supply zones cannot exceed their supply capacities
        for s in range(n_supply_zones):
            model.addCons(
                quicksum(transport_vars[(f"supply_{s}", f"relief_{r}")] for r in range(n_relief_zones)) <= supply_capacities[s], 
                f"Supply_{s}_MaxZoneCapacity"
            )

        # Coverage constraint using Set Covering for Elderly relief zones
        for r in range(n_relief_zones):
            model.addCons(
                quicksum(supply_vars[s] for s in range(n_supply_zones) if distances[s, r] <= self.max_transport_distance) >= 1, 
                f"Relief_{r}_ElderyZoneCoverage"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_supply_zones': 140,
        'n_relief_zones': 67,
        'min_transport_cost': 96,
        'max_transport_cost': 270,
        'min_supply_cost': 1800,
        'max_supply_cost': 2106,
        'min_supply_capacity': 1872,
        'max_supply_capacity': 2500,
        'min_budget_limit': 810,
        'max_budget_limit': 2136,
        'max_transport_distance': 630,
    }

    optimizer = DisasterReliefLogisticsOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")