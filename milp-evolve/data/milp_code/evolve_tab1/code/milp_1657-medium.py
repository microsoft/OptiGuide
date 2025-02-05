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
        transport_costs_land = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_supply_zones, self.n_relief_zones))
        supply_capacities = np.random.randint(self.min_supply_capacity, self.max_supply_capacity + 1, self.n_supply_zones)
        
        # Introducing variability in relief demands
        relief_demand_mean = np.random.gamma(self.demand_shape, self.demand_scale, self.n_relief_zones).astype(int)
        relief_demand_variation = np.random.gamma(self.demand_shape_variation, self.demand_scale_variation, self.n_relief_zones).astype(int)

        G = nx.DiGraph()
        node_pairs = []
        for s in range(self.n_supply_zones):
            for r in range(self.n_relief_zones):
                G.add_edge(f"supply_{s}", f"relief_{r}")
                node_pairs.append((f"supply_{s}", f"relief_{r}"))

        return {
            "supply_costs": supply_costs,
            "transport_costs_land": transport_costs_land,
            "supply_capacities": supply_capacities,
            "relief_demand_mean": relief_demand_mean,
            "relief_demand_variation": relief_demand_variation,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        supply_costs = instance['supply_costs']
        transport_costs_land = instance['transport_costs_land']
        supply_capacities = instance['supply_capacities']
        relief_demand_mean = instance['relief_demand_mean']
        relief_demand_variation = instance['relief_demand_variation']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("DisasterReliefLogisticsOptimization")
        n_supply_zones = len(supply_costs)
        n_relief_zones = len(transport_costs_land[0])
        
        # Decision variables
        supply_vars = {s: model.addVar(vtype="B", name=f"Supply_{s}") for s in range(n_supply_zones)}
        transport_vars_land = {(u, v): model.addVar(vtype="C", name=f"TransportLand_{u}_{v}") for u, v in node_pairs}
        
        # Objective: minimize the total cost including supply zone costs and transport costs.
        model.setObjective(
            quicksum(supply_costs[s] * supply_vars[s] for s in range(n_supply_zones)) +
            quicksum(transport_costs_land[s, int(v.split('_')[1])] * transport_vars_land[(u, v)] for (u, v) in node_pairs for s in range(n_supply_zones) if u == f'supply_{s}'),
            "minimize"
        )

        # New Constraints: Ensure robust total supply meets stochastic demand
        for r in range(n_relief_zones):
            # Robust demand satisfaction with variation
            model.addCons(
                quicksum(transport_vars_land[(u, f"relief_{r}")] for u in G.predecessors(f"relief_{r}")) >= relief_demand_mean[r] + relief_demand_variation[r], 
                f"Relief_{r}_RobustDemandSatisfaction"
            )

        # Constraints: Transport is feasible only if supply zones are operational
        for s in range(n_supply_zones):
            for r in range(n_relief_zones):
                model.addCons(
                    transport_vars_land[(f"supply_{s}", f"relief_{r}")] <= supply_vars[s] * self.max_transport_distance,
                    f"Supply_{s}_SupplyLimitByDistance_{r}"
                )

        # Modified Constraints: Supply zones cannot exceed their capacities considering robustness
        for s in range(n_supply_zones):
            model.addCons(
                quicksum(transport_vars_land[(f"supply_{s}", f"relief_{r}")] for r in range(n_relief_zones)) <= 
                supply_capacities[s] * (1 - self.supply_capacity_variability), 
                f"Supply_{s}_MaxZoneCapacityRobust"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_supply_zones': 200,
        'n_relief_zones': 160,
        'min_transport_cost': 50,
        'max_transport_cost': 300,
        'min_supply_cost': 100,
        'max_supply_cost': 1500,
        'min_supply_capacity': 180,
        'max_supply_capacity': 1000,
        'max_transport_distance': 500,
        'demand_shape': 20.0,
        'demand_scale': 20.0,
        'demand_shape_variation': 2.0,
        'demand_scale_variation': 5.0,
        'supply_capacity_variability': 0.1
    }

    optimizer = DisasterReliefLogisticsOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")