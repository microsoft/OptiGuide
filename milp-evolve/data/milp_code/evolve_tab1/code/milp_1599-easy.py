import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ComplexLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation with more diverse data components
    def generate_instance(self):
        assert self.n_vehicles > 0 and self.n_zones > 0
        assert self.min_fuel_cost >= 0 and self.max_fuel_cost >= self.min_fuel_cost
        
        # Introducing correlation between fuel costs and capacities
        fuel_costs = np.random.uniform(self.min_fuel_cost, self.max_fuel_cost, self.n_vehicles)
        capacities = np.random.gamma(self.capacity_shape, self.capacity_scale, self.n_vehicles).astype(int)
        demands = np.random.normal(self.mean_demands, self.std_dev_demands, self.n_zones).astype(int)
        zones = range(self.n_zones)
        
        # Random distances for each vehicle to each zone
        distances = {(v, z): np.random.randint(1, 100) for v in range(self.n_vehicles) for z in zones}

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost, self.n_vehicles)
        
        return {
            "fuel_costs": fuel_costs,
            "demands": demands,
            "capacities": capacities,
            "distances": distances,
            "fixed_costs": fixed_costs
        }

    # MILP modeling with a more complex objective and constraints
    def solve(self, instance):
        fuel_costs = instance["fuel_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]
        distances = instance["distances"]
        fixed_costs = instance["fixed_costs"]

        model = Model("ComplexLogisticsOptimization")
        n_vehicles = len(capacities)
        n_zones = len(demands)

        # Decision variables
        vehicle_allocated = {v: model.addVar(vtype="B", name=f"Vehicle_{v}") for v in range(n_vehicles)}
        deliveries = {(v, z): model.addVar(vtype="C", name=f"Delivery_{v}_Zone_{z}") for v in range(n_vehicles) for z in range(n_zones)}
        unmet_deliveries = {z: model.addVar(vtype="C", name=f"Unmet_Deliveries_Zone_{z}") for z in range(n_zones)}

        # Aggregated cost components
        total_fuel_cost = quicksum(fuel_costs[v] * vehicle_allocated[v] for v in range(n_vehicles))
        total_fixed_cost = quicksum(fixed_costs[v] * vehicle_allocated[v] for v in range(n_vehicles))
        total_distance_cost = quicksum(distances[v, z] * deliveries[v, z] for v in range(n_vehicles) for z in range(n_zones))
        penalty_unmet_deliveries = self.penalty_unmet_deliveries * quicksum(unmet_deliveries[z] for z in range(n_zones))
        
        # Objective function: Minimize comprehensive cost
        model.setObjective(
            total_fuel_cost + total_fixed_cost + total_distance_cost + penalty_unmet_deliveries,
            "minimize"
        )

        # Constraint: Demand of each zone must be met
        for z in range(n_zones):
            model.addCons(
                quicksum(deliveries[v, z] for v in range(n_vehicles)) + unmet_deliveries[z] >= demands[z],
                f"Zone_{z}_Demands"
            )

        # Constraint: Each vehicle cannot exceed its capacity in deliveries
        for v in range(n_vehicles):
            model.addCons(
                quicksum(deliveries[v, z] for z in range(n_zones)) <= capacities[v], 
                f"Vehicle_{v}_Capacity"
            )

        # Integrated constraint: Only allocated vehicles can be used for deliveries
        for v in range(n_vehicles):
            for z in range(n_zones):
                model.addCons(
                    deliveries[v, z] <= capacities[v] * vehicle_allocated[v],
                    f"Vehicle_{v}_Delivery_{z}"
                )

        # Balance load among vehicles
        avg_capacity_per_vehicle = sum(capacities) / n_vehicles
        for v in range(n_vehicles):
            model.addCons(
                quicksum(deliveries[v, z] for z in range(n_zones)) <= avg_capacity_per_vehicle * 1.2,
                f"Balanced_Vehicle_{v}"
            )

        # Solve the model
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
        'capacity_shape': 5,
        'capacity_scale': 300,
        'min_fixed_cost': 5000,
        'max_fixed_cost': 10000,
        'penalty_unmet_deliveries': 2000,
    }

    optimizer = ComplexLogisticsOptimization(parameters, seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")