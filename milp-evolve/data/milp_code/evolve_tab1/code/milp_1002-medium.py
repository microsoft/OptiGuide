import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class UrbanLogisticsComplex:
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

        # Reduced data for hazardous material handling
        hazard_limits = self.randint(self.n_centers, self.hazard_limit_interval)
        handling_costs = self.randint(self.n_centers, self.handling_cost_interval)
        time_windows = self.randint(self.n_trucks, self.delivery_window_interval)

        # Hazard zones - randomly decide which centers can handle hazards
        hazard_zones = np.random.choice([0, 1], size=self.n_centers, p=[0.7, 0.3])
        hazard_eligible_trucks = np.random.choice([0, 1], size=self.n_trucks, p=[0.8, 0.2])

        # Create a subgraph (similar to coverage zones) for centers
        G = nx.barabasi_albert_graph(self.n_centers, 3)
        coverage_zones = list(nx.find_cliques(G))

        res = {
            'truck_arrival_rates': truck_arrival_rates,
            'center_capacities': center_capacities,
            'activation_costs': activation_costs,
            'truck_travel_times': truck_travel_times,
            'hazard_limits': hazard_limits,
            'handling_costs': handling_costs,
            'time_windows': time_windows,
            'hazard_zones': hazard_zones,
            'hazard_eligible_trucks': hazard_eligible_trucks,
            'coverage_zones': coverage_zones
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
        time_windows = instance['time_windows']
        hazard_zones = instance['hazard_zones']
        hazard_eligible_trucks = instance['hazard_eligible_trucks']
        coverage_zones = instance['coverage_zones']

        n_trucks = len(truck_arrival_rates)
        n_centers = len(center_capacities)

        model = Model("UrbanLogisticsComplex")

        # Decision variables
        activate_center = {m: model.addVar(vtype="B", name=f"Activate_{m}") for m in range(n_centers)}
        allocate_truck = {(n, m): model.addVar(vtype="B", name=f"Allocate_{n}_{m}") for n in range(n_trucks) for m in range(n_centers)}
        hazard_use = {m: model.addVar(vtype="B", name=f"HazardUse_{m}") for m in range(n_centers)}

        # Time variables for delivery windows
        delivery_time = {(n, m): model.addVar(vtype="C", name=f"DeliveryTime_{n}_{m}") for n in range(n_trucks) for m in range(n_centers)}

        # Maintenance variable
        maintenance_budget = model.addVar(vtype="C", name="Maintenance_Budget")

        # Objective: Minimize the total cost including activation, travel time penalty, handling cost, hazardous material usage, and maintenance
        penalty_per_travel_time = 100
        penalty_per_maintenance = 50
        
        objective_expr = quicksum(activation_costs[m] * activate_center[m] for m in range(n_centers)) + \
                         penalty_per_travel_time * quicksum(truck_travel_times[n, m] * allocate_truck[n, m] for n in range(n_trucks) for m in range(n_centers)) + \
                         quicksum(handling_costs[m] * hazard_use[m] for m in range(n_centers)) + \
                         penalty_per_maintenance * maintenance_budget

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

        # Hazardous material handling must be within limits
        for m in range(n_centers):
            model.addCons(hazard_use[m] <= hazard_limits[m], f"Hazard_Use_Limit_{m}")
            model.addCons(hazard_use[m] <= hazard_zones[m], f"Hazard_Zone_Limit_{m}")

        # Delivery must be within the time window
        for n in range(n_trucks):
            for m in range(n_centers):
                model.addCons(delivery_time[n, m] == time_windows[n], f"Delivery_Time_Window_{n}_{m}")

        # Ensure only hazard-eligible trucks are allocated to centers handling hazardous materials
        for n in range(n_trucks):
            for m in range(n_centers):
                model.addCons(allocate_truck[n, m] <= hazard_eligible_trucks[n] + (1 - hazard_use[m]), f"Hazard_Eligible_{n}_{m}")

        # Ensure only one center is activated in a coverage zone
        for i, zone in enumerate(coverage_zones):
            model.addCons(quicksum(activate_center[m] for m in zone) <= 1, f"Coverage_Zone_{i}")

        model.addCons(maintenance_budget <= sum(time_windows) / 3.0, name="Maintenance_Budget_Limit")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_trucks': 63,
        'n_centers': 148,
        'arrival_rate_interval': (168, 1687),
        'capacity_interval': (379, 1138),
        'activation_cost_interval': (843, 1686),
        'ratio': 2126.25,
        'hazard_limit_interval': (18, 93),
        'handling_cost_interval': (278, 1138),
        'delivery_window_interval': (404, 1620),
    }
    
    urban_logistics_complex = UrbanLogisticsComplex(parameters, seed=seed)
    instance = urban_logistics_complex.generate_instance()
    solve_status, solve_time = urban_logistics_complex.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")