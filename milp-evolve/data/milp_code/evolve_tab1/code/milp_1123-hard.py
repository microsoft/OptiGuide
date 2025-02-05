import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class DeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.num_locations > 1 and self.num_vehicles > 0
        assert self.min_distance >= 0 and self.max_distance >= self.min_distance
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        assert self.min_fuel_consumption > 0 and self.max_fuel_consumption >= self.min_fuel_consumption
        assert self.min_emission > 0 and self.max_emission >= self.min_emission
        
        travel_distances = np.random.randint(self.min_distance, self.max_distance + 1, (self.num_locations, self.num_locations))
        vehicle_capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.num_vehicles)
        fuel_consumptions = np.random.uniform(self.min_fuel_consumption, self.max_fuel_consumption, (self.num_vehicles, self.num_locations))
        emissions = np.random.uniform(self.min_emission, self.max_emission, (self.num_vehicles, self.num_locations))

        demands = np.random.randint(1, 10, self.num_locations)
        # Setting the depot demand to 0
        demands[0] = 0
        
        return {
            "travel_distances": travel_distances,
            "vehicle_capacities": vehicle_capacities,
            "fuel_consumptions": fuel_consumptions,
            "emissions": emissions,
            "demands": demands
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        travel_distances = instance['travel_distances']
        vehicle_capacities = instance['vehicle_capacities']
        fuel_consumptions = instance['fuel_consumptions']
        emissions = instance['emissions']
        demands = instance['demands']

        model = Model("DeliveryOptimization")
        num_locations = len(demands)
        num_vehicles = len(vehicle_capacities)
        
        # Decision variables
        route_vars = {(i, j, k): model.addVar(vtype="B", name=f"Route_{i}_{j}_Veh_{k}")
                      for i in range(num_locations) for j in range(num_locations) for k in range(num_vehicles) if i != j}
        fuel_vars = {k: model.addVar(vtype="C", lb=0, name=f"Fuel_Veh_{k}") for k in range(num_vehicles)}
        emission_vars = {k: model.addVar(vtype="C", lb=0, name=f"Emission_Veh_{k}") for k in range(num_vehicles)}

        # Objective: Minimize travel distances, fuel consumption and emissions
        model.setObjective(
            quicksum(travel_distances[i, j] * route_vars[i, j, k] for i in range(num_locations) for j in range(num_locations) for k in range(num_vehicles) if i != j) +
            quicksum(fuel_vars[k] for k in range(num_vehicles)) +
            quicksum(emission_vars[k] for k in range(num_vehicles)),
            "minimize"
        )
        
        # Constraints: Each location must be visited exactly once
        for j in range(1, num_locations):
            model.addCons(quicksum(route_vars[i, j, k] for i in range(num_locations) for k in range(num_vehicles) if i != j) == 1, f"Visit_{j}")

        # Constraints: Leaving a location for each vehicle must match the number of visits
        for k in range(num_vehicles):
            for i in range(num_locations):
                model.addCons(quicksum(route_vars[i, j, k] for j in range(num_locations) if i != j) ==
                              quicksum(route_vars[j, i, k] for j in range(num_locations) if i != j), f"FlowConservation_Veh_{k}_Loc_{i}")
                
        # Constraints: Vehicle capacity
        for k in range(num_vehicles):
            model.addCons(
                quicksum(demands[j] * quicksum(route_vars[i, j, k] for i in range(num_locations) if i != j) for j in range(1, num_locations)) <= vehicle_capacities[k],
                f"Capacity_Veh_{k}"
            )
        
        # Fuel consumption constraints
        for k in range(num_vehicles):
            model.addCons(
                fuel_vars[k] == quicksum(fuel_consumptions[k, j] * quicksum(route_vars[i, j, k] for i in range(num_locations) if i != j) for j in range(1, num_locations)),
                f"FuelConsumption_Veh_{k}"
            )
        
        # Emission constraints
        for k in range(num_vehicles):
            model.addCons(
                emission_vars[k] == quicksum(emissions[k, j] * quicksum(route_vars[i, j, k] for i in range(num_locations) if i != j) for j in range(1, num_locations)),
                f"Emission_Veh_{k}"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_locations': 37,
        'num_vehicles': 5,
        'min_distance': 7,
        'max_distance': 1500,
        'min_capacity': 50,
        'max_capacity': 50,
        'min_fuel_consumption': 0.1,
        'max_fuel_consumption': 0.31,
        'min_emission': 0.38,
        'max_emission': 0.73,
    }
    delivery_optimizer = DeliveryOptimization(parameters, seed=42)
    instance = delivery_optimizer.generate_instance()
    solve_status, solve_time, objective_value = delivery_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")