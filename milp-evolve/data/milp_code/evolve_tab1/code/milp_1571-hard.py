import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_vehicles > 0 and self.n_zones > 0
        assert self.min_fuel_cost >= 0 and self.max_fuel_cost >= self.min_fuel_cost

        fuel_costs = np.random.randint(self.min_fuel_cost, self.max_fuel_cost + 1, self.n_vehicles)
        demands = np.random.normal(self.mean_demands, self.std_dev_demands, self.n_zones).astype(int)
        capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity + 1, self.n_vehicles)

        return {
            "fuel_costs": fuel_costs,
            "demands": demands,
            "capacities": capacities
        }

    # MILP modeling
    def solve(self, instance):
        fuel_costs = instance["fuel_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]

        model = Model("SimplifiedLogisticsOptimization")
        n_vehicles = len(capacities)
        n_zones = len(demands)

        # Decision variables
        vehicle_allocated = {v: model.addVar(vtype="B", name=f"Vehicle_{v}") for v in range(n_vehicles)}
        deliveries = {(v, z): model.addVar(vtype="C", name=f"Delivery_{v}_Zone_{z}") for v in range(n_vehicles) for z in range(n_zones)}
        unmet_deliveries = {z: model.addVar(vtype="C", name=f"Unmet_Deliveries_Zone_{z}") for z in range(n_zones)}

        # Objective function: Minimize total cost
        model.setObjective(
            quicksum(fuel_costs[v] * vehicle_allocated[v] for v in range(n_vehicles)) +
            self.penalty_unmet_deliveries * quicksum(unmet_deliveries[z] for z in range(n_zones)),
            "minimize"
        )

        # Constraints: Each zone must meet its delivery demands
        for z in range(n_zones):
            model.addCons(
                quicksum(deliveries[v, z] for v in range(n_vehicles)) + unmet_deliveries[z] >= demands[z],
                f"Zone_{z}_Demands"
            )

        # Constraints: Each vehicle cannot exceed its capacity in deliveries
        for v in range(n_vehicles):
            model.addCons(
                quicksum(deliveries[v, z] for z in range(n_zones)) <= capacities[v], 
                f"Vehicle_{v}_Capacity"
            )

        # Constraints: Only allocated vehicles can be used for deliveries
        for v in range(n_vehicles):
            for z in range(n_zones):
                model.addCons(
                    deliveries[v, z] <= capacities[v] * vehicle_allocated[v],
                    f"Vehicle_{v}_Delivery_{z}"
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
        'n_vehicles': 150,
        'n_zones': 180,
        'min_fuel_cost': 1350,
        'max_fuel_cost': 1800,
        'mean_demands': 2000,
        'std_dev_demands': 3000,
        'min_vehicle_capacity': 2000,
        'max_vehicle_capacity': 3000,
        'penalty_unmet_deliveries': 2000,
    }

    optimizer = SimplifiedLogisticsOptimization(parameters, seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")