import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_zones > 0 and self.n_vehicles > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_supply_cost >= 0 and self.max_supply_cost >= self.min_supply_cost

        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, self.n_vehicles)
        supply_costs = np.random.randint(self.min_supply_cost, self.max_supply_cost + 1, self.n_vehicles)
        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_zones).astype(int)
        capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity + 1, self.n_vehicles)

        return {
            "transport_costs": transport_costs,
            "supply_costs": supply_costs,
            "demands": demands,
            "capacities": capacities
        }

    # MILP modeling
    def solve(self, instance):
        transport_costs = instance["transport_costs"]
        supply_costs = instance["supply_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]

        model = Model("SupplyDistributionOptimization")
        n_vehicles = len(capacities)
        n_zones = len(demands)

        # Decision variables
        vehicle_used = {v: model.addVar(vtype="B", name=f"Vehicle_{v}") for v in range(n_vehicles)}
        supply_delivery = {(v, z): model.addVar(vtype="C", name=f"Vehicle_{v}_Zone_{z}") for v in range(n_vehicles) for z in range(n_zones)}
        unmet_demand = {z: model.addVar(vtype="C", name=f"Unmet_Demand_Zone_{z}") for z in range(n_zones)}

        # Objective function: Minimize total cost
        model.setObjective(
            quicksum(transport_costs[v] * vehicle_used[v] for v in range(n_vehicles)) +
            quicksum(supply_costs[v] * supply_delivery[v, z] for v in range(n_vehicles) for z in range(n_zones)) +
            self.penalty_unmet_demand * quicksum(unmet_demand[z] for z in range(n_zones)),
            "minimize"
        )

        # Constraints: Each zone must meet its demand
        for z in range(n_zones):
            model.addCons(
                quicksum(supply_delivery[v, z] for v in range(n_vehicles)) + unmet_demand[z] >= demands[z],
                f"Zone_{z}_Demand"
            )

        # Constraints: Each vehicle cannot exceed its capacity
        for v in range(n_vehicles):
            model.addCons(
                quicksum(supply_delivery[v, z] for z in range(n_zones)) <= capacities[v], 
                f"Vehicle_{v}_Capacity"
            )

        # Constraints: Only used vehicles can make deliveries
        for v in range(n_vehicles):
            for z in range(n_zones):
                model.addCons(
                    supply_delivery[v, z] <= capacities[v] * vehicle_used[v],
                    f"Vehicle_{v}_Deliver_{z}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value

if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_vehicles': 700,
        'n_zones': 150,
        'min_transport_cost': 15,
        'max_transport_cost': 1500,
        'mean_demand': 400,
        'std_dev_demand': 1500,
        'min_vehicle_capacity': 1000,
        'max_vehicle_capacity': 3500,
        'min_supply_cost': 400,
        'max_supply_cost': 2500,
        'penalty_unmet_demand': 1000,
    }

    supply_optimizer = SupplyDistributionOptimization(parameters, seed)
    instance = supply_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")