import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedDisasterReliefLogisticsOptimization:
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
        relief_demands = np.random.gamma(self.demand_shape, self.demand_scale, self.n_relief_zones).astype(int)

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
            "relief_demands": relief_demands,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        supply_costs = instance['supply_costs']
        transport_costs_land = instance['transport_costs_land']
        supply_capacities = instance['supply_capacities']
        relief_demands = instance['relief_demands']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("SimplifiedDisasterReliefLogisticsOptimization")
        n_supply_zones = len(supply_costs)
        n_relief_zones = len(transport_costs_land[0])
        
        # Decision variables
        supply_vars = {s: model.addVar(vtype="B", name=f"Supply_{s}") for s in range(n_supply_zones)}
        transport_vars_land = {(u, v): model.addVar(vtype="C", name=f"TransportLand_{u}_{v}") for u, v in node_pairs}
        unmet_demand_vars = {r: model.addVar(vtype="C", name=f"UnmetDemand_{r}") for r in range(n_relief_zones)}

        # Objective: minimize the total cost including supply zone costs, transport costs, and unmet demand penalties.
        penalty_per_unit_unmet_demand = 1000
        model.setObjective(
            quicksum(supply_costs[s] * supply_vars[s] for s in range(n_supply_zones)) +
            quicksum(transport_costs_land[s, int(v.split('_')[1])] * transport_vars_land[(u, v)] for (u, v) in node_pairs for s in range(n_supply_zones) if u == f'supply_{s}') +
            penalty_per_unit_unmet_demand * quicksum(unmet_demand_vars[r] for r in range(n_relief_zones)),
            "minimize"
        )

        # Constraints: Ensure total supply matches demand accounting for unmet demand
        for r in range(n_relief_zones):
            model.addCons(
                quicksum(transport_vars_land[(u, f"relief_{r}")] for u in G.predecessors(f"relief_{r}")) + unmet_demand_vars[r] >= relief_demands[r], 
                f"Relief_{r}_DemandSatisfaction"
            )

        # Constraints: Transport is feasible only if supply zones are operational
        for s in range(n_supply_zones):
            for r in range(n_relief_zones):
                model.addCons(
                    transport_vars_land[(f"supply_{s}", f"relief_{r}")] <= supply_vars[s] * self.max_transport_distance,
                    f"Supply_{s}_SupplyLimitByDistance_{r}"
                )

        # Constraints: Supply zones cannot exceed their capacities
        for s in range(n_supply_zones):
            model.addCons(
                quicksum(transport_vars_land[(f"supply_{s}", f"relief_{r}")] for r in range(n_relief_zones)) <= supply_capacities[s], 
                f"Supply_{s}_MaxZoneCapacity"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_supply_zones': 1400,
        'n_relief_zones': 32,
        'min_transport_cost': 144,
        'max_transport_cost': 1080,
        'min_supply_cost': 1350,
        'max_supply_cost': 1579,
        'min_supply_capacity': 1122,
        'max_supply_capacity': 2250,
        'max_transport_distance': 2000,
        'demand_shape': 54.0,
        'demand_scale': 0.2,
    }

    optimizer = SimplifiedDisasterReliefLogisticsOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")