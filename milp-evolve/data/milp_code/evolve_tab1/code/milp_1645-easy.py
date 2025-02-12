import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class RenewableEnergyOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_energy_farms > 0 and self.n_zones > 0
        assert self.min_farm_cost >= 0 and self.max_farm_cost >= self.min_farm_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_farm_capacity > 0 and self.max_farm_capacity >= self.min_farm_capacity

        farm_costs = np.random.randint(self.min_farm_cost, self.max_farm_cost + 1, self.n_energy_farms)
        transport_costs_vehicle = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_energy_farms, self.n_zones))
        farm_capacities = np.random.randint(self.min_farm_capacity, self.max_farm_capacity + 1, self.n_energy_farms)
        energy_demand = np.random.normal(self.demand_mean, self.demand_stddev, self.n_zones).astype(int)

        G = nx.DiGraph()
        node_pairs = []
        for f in range(self.n_energy_farms):
            for z in range(self.n_zones):
                G.add_edge(f"energy_farm_{f}", f"zone_{z}")
                node_pairs.append((f"energy_farm_{f}", f"zone_{z}"))

        return {
            "farm_costs": farm_costs,
            "transport_costs_vehicle": transport_costs_vehicle,
            "farm_capacities": farm_capacities,
            "energy_demand": energy_demand,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        farm_costs = instance['farm_costs']
        transport_costs_vehicle = instance['transport_costs_vehicle']
        farm_capacities = instance['farm_capacities']
        energy_demand = instance['energy_demand']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("RenewableEnergyOptimization")
        n_energy_farms = len(farm_costs)
        n_zones = len(transport_costs_vehicle[0])
        
        # Decision variables
        energy_farm_vars = {f: model.addVar(vtype="B", name=f"EnergyFarm_{f}") for f in range(n_energy_farms)}
        vehicle_transport_vars = {(u, v): model.addVar(vtype="C", name=f"VehicleTransport_{u}_{v}") for u, v in node_pairs}
        unmet_energy_storage_vars = {z: model.addVar(vtype="C", name=f"UnmetEnergyStorage_{z}") for z in range(n_zones)}

        # Objective: minimize the total cost including energy farm costs, transport costs, and unmet energy storage penalties.
        penalty_per_unit_unmet_energy = 1500
        model.setObjective(
            quicksum(farm_costs[f] * energy_farm_vars[f] for f in range(n_energy_farms)) +
            quicksum(transport_costs_vehicle[f, int(v.split('_')[1])] * vehicle_transport_vars[(u, v)] for (u, v) in node_pairs for f in range(n_energy_farms) if u == f'energy_farm_{f}') +
            penalty_per_unit_unmet_energy * quicksum(unmet_energy_storage_vars[z] for z in range(n_zones)),
            "minimize"
        )

        # Constraints: Ensure total energy supply matches demand accounting for unmet demand
        for z in range(n_zones):
            model.addCons(
                quicksum(vehicle_transport_vars[(u, f"zone_{z}")] for u in G.predecessors(f"zone_{z}")) + unmet_energy_storage_vars[z] >= energy_demand[z], 
                f"Zone_{z}_EnergyDemand"
            )

        # Constraints: Transport is feasible only if energy farms are operational and within maximum transport time
        for f in range(n_energy_farms):
            for z in range(n_zones):
                model.addCons(
                    vehicle_transport_vars[(f"energy_farm_{f}", f"zone_{z}")] <= energy_farm_vars[f] * self.max_transport_time,
                    f"EnergyFarm_{f}_MaxTransportTime_{z}"
                )

        # Constraints: Energy farms' distribution should not exceed their capacity
        for f in range(n_energy_farms):
            model.addCons(
                quicksum(vehicle_transport_vars[(f"energy_farm_{f}", f"zone_{z}")] for z in range(n_zones)) <= farm_capacities[f], 
                f"EnergyFarm_{f}_MaxFarmCapacity"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_energy_farms': 2800,
        'n_zones': 32,
        'min_transport_cost': 1296,
        'max_transport_cost': 2430,
        'min_farm_cost': 1012,
        'max_farm_cost': 1579,
        'min_farm_capacity': 2244,
        'max_farm_capacity': 2250,
        'max_transport_time': 1000,
        'demand_mean': 1000,
        'demand_stddev': 500,
    }

    optimizer = RenewableEnergyOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")