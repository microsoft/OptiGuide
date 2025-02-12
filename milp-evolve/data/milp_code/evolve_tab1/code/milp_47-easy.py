import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class CapacitatedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_facilities, self.fixed_cost_cste_interval)
        )
        travel_times = self.unit_transportation_costs() * demands[:, np.newaxis]

        time_windows = np.column_stack((
            self.randint(self.n_customers, self.start_time_window_interval),
            self.randint(self.n_customers, self.end_time_window_interval)
        ))

        vehicle_types = np.random.choice(["Type1", "Type2", "Type3"], size=self.n_facilities, p=[0.4, 0.4, 0.2])
        type_capacities = {"Type1": 50, "Type2": 100, "Type3": 150}
        capacities = np.array([type_capacities[v] for v in vehicle_types])
        max_load = self.max_vehicle_capacity

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'travel_times': travel_times,
            'time_windows': time_windows,
            'max_load': max_load
        }
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        travel_times = instance['travel_times']
        time_windows = instance['time_windows']
        max_load = instance['max_load']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("CapacitatedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        
        start_times = {i: model.addVar(vtype="C", name=f"Start_{i}") for i in range(n_customers)}    
        
        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(travel_times[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        
        # Constraints: demand must be met
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        # Constraints: tightening constraints
        total_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, "TotalDemand")

        # Constraints: time windows
        for i in range(n_customers):
            model.addCons(start_times[i] >= time_windows[i, 0], f"StartAfter_{i}")
            model.addCons(start_times[i] <= time_windows[i, 1], f"EndBefore_{i}")

        # Constraints: arrival times based on traffic conditions
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}")
                model.addCons(start_times[i] + travel_times[i, j] >= start_times[j], f"TrafficConditions_{i}_{j}")

        # Constraints: vehicle load utilization
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= max_load, f"MaxLoad_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 100,
        'n_facilities': 100,
        'demand_interval': (5, 36),
        'capacity_interval': (10, 161),
        'fixed_cost_scale_interval': (100, 111),
        'fixed_cost_cste_interval': (0, 91),
        'start_time_window_interval': (0, 10),
        'end_time_window_interval': (10, 30),
        'ratio': 5.0,
        'max_vehicle_capacity': 150,
        'continuous_assignment': True,
    }

    facility_location = CapacitatedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")