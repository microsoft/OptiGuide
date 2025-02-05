import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

class Fleet:
    def __init__(self, number_of_vehicles, vehicle_capacities, travel_times, demands):
        self.number_of_vehicles = number_of_vehicles
        self.vehicle_capacities = vehicle_capacities
        self.travel_times = travel_times
        self.demands = demands

class LogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.num_vehicles > 0 and self.num_tasks > 0
        assert self.min_capacity >= 0 and self.max_capacity >= self.min_capacity
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand

        vehicle_capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.num_vehicles)
        demands = np.random.randint(self.min_demand, self.max_demand + 1, self.num_tasks)
        travel_times = np.random.randint(self.min_travel_time, self.max_travel_time + 1, (self.num_vehicles, self.num_tasks))
        operating_costs = np.random.randint(self.min_operating_cost, self.max_operating_cost + 1, self.num_vehicles)
        new_route_penalties = np.random.randint(self.min_new_route_penalty, self.max_new_route_penalty + 1, self.num_tasks)
        
        return {
            "vehicle_capacities": vehicle_capacities,
            "demands": demands,
            "travel_times": travel_times,
            "operating_costs": operating_costs,
            "new_route_penalties": new_route_penalties
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        vehicle_capacities = instance['vehicle_capacities']
        demands = instance['demands']
        travel_times = instance['travel_times']
        operating_costs = instance['operating_costs']
        new_route_penalties = instance['new_route_penalties']
        
        model = Model("LogisticsOptimization")
        num_vehicles = len(vehicle_capacities)
        num_tasks = len(demands)

        # Decision variables
        Assignment_vars = {(v, t): model.addVar(vtype="B", name=f"Assignment_{v}_{t}") for v in range(num_vehicles) for t in range(num_tasks)}
        NewRoutes_vars = {t: model.addVar(vtype="B", name=f"NewRoute_{t}") for t in range(num_tasks)}
        TravelTime_vars = {(v, t): model.addVar(vtype="C", name=f"TravelTime_{v}_{t}") for v in range(num_vehicles) for t in range(num_tasks)}

        # Objective: minimize the total travel time and operating costs
        model.setObjective(
            quicksum(travel_times[v, t] * Assignment_vars[v, t] for v in range(num_vehicles) for t in range(num_tasks)) +
            quicksum(operating_costs[v] * quicksum(Assignment_vars[v, t] for t in range(num_tasks)) for v in range(num_vehicles)) +
            quicksum(new_route_penalties[t] * NewRoutes_vars[t] for t in range(num_tasks)), "minimize"
        )

        # Constraints: Each task must be assigned to exactly one vehicle or marked as a new route
        for t in range(num_tasks):
            model.addCons(quicksum(Assignment_vars[v, t] for v in range(num_vehicles)) + NewRoutes_vars[t] == 1, f"TaskAssignment_{t}")

        # Constraints: Vehicle capacity constraints
        for v in range(num_vehicles):
            model.addCons(quicksum(demands[t] * Assignment_vars[v, t] for t in range(num_tasks)) <= vehicle_capacities[v], f"VehicleCapacity_{v}")

        # Constraints: Travel time consistency
        M = self.M  # A large constant
        for v in range(num_vehicles):
            for t in range(num_tasks):
                model.addCons(TravelTime_vars[v, t] == travel_times[v, t] * Assignment_vars[v, t], f"TravelTimeConsistency_{v}_{t}")
                model.addCons(TravelTime_vars[v, t] <= M * Assignment_vars[v, t], f"BigM_TravelTime_{v}_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_vehicles': 40,
        'num_tasks': 200,
        'min_capacity': 50,
        'max_capacity': 600,
        'min_demand': 50,
        'max_demand': 180,
        'min_travel_time': 7,
        'max_travel_time': 250,
        'min_operating_cost': 100,
        'max_operating_cost': 1000,
        'min_new_route_penalty': 500,
        'max_new_route_penalty': 600,
        'M': 2000,
    }

    logistics_optimizer = LogisticsOptimization(parameters, seed)
    instance = logistics_optimizer.generate_instance()
    solve_status, solve_time, objective_value = logistics_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")