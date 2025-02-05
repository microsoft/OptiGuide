import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FleetAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_trucks > 0 and self.n_destinations > 0
        assert self.min_fuel_cost >= 0 and self.max_fuel_cost >= self.min_fuel_cost
        assert self.min_truck_capacity > 0 and self.max_truck_capacity >= self.min_truck_capacity
        assert self.min_travel_time >= 0 and self.max_travel_time >= self.min_travel_time

        fuel_costs = np.random.randint(self.min_fuel_cost, self.max_fuel_cost + 1, (self.n_trucks, self.n_destinations))
        capacities = np.random.randint(self.min_truck_capacity, self.max_truck_capacity + 1, self.n_trucks)
        travel_times = np.random.randint(self.min_travel_time, self.max_travel_time + 1, (self.n_trucks, self.n_destinations))
        arrival_times = np.random.randint(self.min_arrival_time, self.max_arrival_time + 1, self.n_destinations)
        
        return {
            "fuel_costs": fuel_costs,
            "capacities": capacities,
            "travel_times": travel_times,
            "arrival_times": arrival_times,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        fuel_costs = instance['fuel_costs']
        capacities = instance['capacities']
        travel_times = instance['travel_times']
        arrival_times = instance['arrival_times']
        
        model = Model("FleetAllocation")
        n_trucks = len(fuel_costs)
        n_destinations = len(fuel_costs[0])
        
        # Decision variables
        truck_vars = {t: model.addVar(vtype="B", name=f"Truck_{t}") for t in range(n_trucks)}
        allocation_vars = {(t, d): model.addVar(vtype="B", name=f"Truck_{t}_Destination_{d}") for t in range(n_trucks) for d in range(n_destinations)}
        
        # Objective: minimize the total fuel and travel time cost
        model.setObjective(quicksum(fuel_costs[t][d] * allocation_vars[t, d] for t in range(n_trucks) for d in range(n_destinations)) +
                           quicksum(travel_times[t][d] * allocation_vars[t, d] for t in range(n_trucks) for d in range(n_destinations)), "minimize")
        
        # Constraints: Each destination is served by exactly one truck
        for d in range(n_destinations):
            model.addCons(quicksum(allocation_vars[t, d] for t in range(n_trucks)) == 1, f"Destination_{d}_Assignment")
        
        # Constraints: Only used trucks can serve destinations
        for t in range(n_trucks):
            for d in range(n_destinations):
                model.addCons(allocation_vars[t, d] <= truck_vars[t], f"Truck_{t}_Service_{d}")
        
        # Constraints: Trucks cannot exceed their capacity
        for t in range(n_trucks):
            model.addCons(quicksum(allocation_vars[t, d] * arrival_times[d] for d in range(n_destinations)) <= capacities[t], f"Truck_{t}_Capacity")
        
        # Constraints: Travel time should respect maximum hours for each truck
        for t in range(n_trucks):
            model.addCons(quicksum(travel_times[t][d] * allocation_vars[t, d] for d in range(n_destinations)) <= self.max_travel_time, f"Truck_{t}_MaxTime")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
        
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_trucks': 80,
        'n_destinations': 200,
        'min_fuel_cost': 500,
        'max_fuel_cost': 2500,
        'min_truck_capacity': 40,
        'max_truck_capacity': 700,
        'min_travel_time': 8,
        'max_travel_time': 40,
        'min_arrival_time': 35,
        'max_arrival_time': 120,
    }

    fleet_optimizer = FleetAllocation(parameters, seed=42)
    instance = fleet_optimizer.generate_instance()
    solve_status, solve_time, objective_value = fleet_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")