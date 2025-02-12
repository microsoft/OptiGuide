import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class UrbanLogistics:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def truck_travel_times(self):
        base_travel_time = 20.0  # base travel time in minutes
        return base_travel_time * np.random.rand(self.n_trucks, self.n_centers)

    def generate_instance(self):
        truck_arrival_rates = self.randint(self.n_trucks, self.arrival_rate_interval)
        center_capacities = self.randint(self.n_centers, self.capacity_interval)
        activation_costs = self.randint(self.n_centers, self.activation_cost_interval)
        truck_travel_times = self.truck_travel_times()

        center_capacities = center_capacities * self.ratio * np.sum(truck_arrival_rates) / np.sum(center_capacities)
        center_capacities = np.round(center_capacities)
        
        # New data for resources and hazardous material handling costs
        hazard_limits = self.randint(self.n_centers, self.hazard_limit_interval)
        handling_costs = self.randint(self.n_centers, self.handling_cost_interval)

        res = {
            'truck_arrival_rates': truck_arrival_rates,
            'center_capacities': center_capacities,
            'activation_costs': activation_costs,
            'truck_travel_times': truck_travel_times,
            'hazard_limits': hazard_limits,
            'handling_costs': handling_costs
        }
        
        return res

    def solve(self, instance):
        # Instance data
        truck_arrival_rates = instance['truck_arrival_rates']
        center_capacities = instance['center_capacities']
        activation_costs = instance['activation_costs']
        truck_travel_times = instance['truck_travel_times']
        hazard_limits = instance['hazard_limits']
        handling_costs = instance['handling_costs']

        n_trucks = len(truck_arrival_rates)
        n_centers = len(center_capacities)

        model = Model("UrbanLogistics")

        # Decision variables
        activate_center = {m: model.addVar(vtype="B", name=f"Activate_{m}") for m in range(n_centers)}
        allocate_truck = {(n, m): model.addVar(vtype="B", name=f"Allocate_{n}_{m}") for n in range(n_trucks) for m in range(n_centers)}
        hazard_use = {m: model.addVar(vtype="B", name=f"HazardUse_{m}") for m in range(n_centers)}

        # Objective: Minimize the total cost including activation, travel time penalty, and handling cost
        penalty_per_travel_time = 100
        objective_expr = quicksum(activation_costs[m] * activate_center[m] for m in range(n_centers)) + \
                         penalty_per_travel_time * quicksum(truck_travel_times[n, m] * allocate_truck[n, m] for n in range(n_trucks) for m in range(n_centers)) + \
                         quicksum(handling_costs[m] * hazard_use[m] for m in range(n_centers))

        # Constraints: each truck must be allocated to exactly one center
        for n in range(n_trucks):
            model.addCons(quicksum(allocate_truck[n, m] for m in range(n_centers)) == 1, f"Truck_Allocation_{n}")
        
        # Constraints: center capacity limits must be respected
        for m in range(n_centers):
            model.addCons(quicksum(truck_arrival_rates[n] * allocate_truck[n, m] for n in range(n_trucks)) <= center_capacities[m] * activate_center[m], f"Center_Capacity_{m}")

        # Constraint: Travel times minimized (All trucks reach centers within permissible limits)
        for n in range(n_trucks):
            for m in range(n_centers):
                model.addCons(truck_travel_times[n, m] * allocate_truck[n, m] <= activate_center[m] * 180, f"Travel_Time_Limit_{n}_{m}")

        # New Constraints: Hazardous material handling must be within limits
        for m in range(n_centers):
            model.addCons(hazard_use[m] <= hazard_limits[m], f"Hazard_Use_Limit_{m}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_trucks': 150,
        'n_centers': 74,
        'arrival_rate_interval': (90, 900),
        'capacity_interval': (675, 2025),
        'activation_cost_interval': (750, 1500),
        'ratio': 157.5,
        'hazard_limit_interval': (5, 25),
        'handling_cost_interval': (333, 1350),
    }

    urban_logistics = UrbanLogistics(parameters, seed=seed)
    instance = urban_logistics.generate_instance()
    solve_status, solve_time = urban_logistics.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")