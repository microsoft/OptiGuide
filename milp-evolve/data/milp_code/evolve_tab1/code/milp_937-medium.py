import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class UrbanLogisticsAdvanced:
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
        
        time_windows = self.randint(self.n_trucks, self.delivery_window_interval)

        # New data for bid allocation and drone optimization
        bids = self.generate_bids()
        drone_max_battery = self.randint(self.n_drones, self.drone_battery_interval)
        drone_battery_cost = np.random.gamma(shape=2.0, scale=1.0, size=self.n_drones).tolist()
        charging_station_cost = self.randint(self.charging_station_max_count, self.charging_station_cost_interval)

        res = {
            'truck_arrival_rates': truck_arrival_rates,
            'center_capacities': center_capacities,
            'activation_costs': activation_costs,
            'truck_travel_times': truck_travel_times,
            'hazard_limits': hazard_limits,
            'handling_costs': handling_costs,
            'time_windows': time_windows,
            'bids': bids,
            'drone_max_battery': drone_max_battery,
            'drone_battery_cost': drone_battery_cost,
            'charging_station_cost': charging_station_cost
        }
        
        return res

    def generate_bids(self):
        bids = []
        for _ in range(self.n_bids):
            bid_size = np.random.randint(1, self.max_bundle_size + 1)
            items = np.random.choice(self.n_trucks, size=bid_size, replace=False).tolist()
            price = np.random.uniform(self.min_bid_price, self.max_bid_price)
            bids.append((items, price))
        return bids

    def solve(self, instance):
        # Instance data
        truck_arrival_rates = instance['truck_arrival_rates']
        center_capacities = instance['center_capacities']
        activation_costs = instance['activation_costs']
        truck_travel_times = instance['truck_travel_times']
        hazard_limits = instance['hazard_limits']
        handling_costs = instance['handling_costs']
        time_windows = instance['time_windows']
        bids = instance['bids']
        drone_max_battery = instance['drone_max_battery']
        drone_battery_cost = instance['drone_battery_cost']
        charging_station_cost = instance['charging_station_cost']

        n_trucks = len(truck_arrival_rates)
        n_centers = len(center_capacities)
        n_bids = len(bids)

        model = Model("UrbanLogisticsAdvanced")

        # Decision variables
        activate_center = {m: model.addVar(vtype="B", name=f"Activate_{m}") for m in range(n_centers)}
        allocate_truck = {(n, m): model.addVar(vtype="B", name=f"Allocate_{n}_{m}") for n in range(n_trucks) for m in range(n_centers)}
        hazard_use = {m: model.addVar(vtype="B", name=f"HazardUse_{m}") for m in range(n_centers)}

        # New Variables for Environmental Impact
        environmental_penalty = {m: model.addVar(vtype="C", name=f"Environmental_Penalty_{m}") for m in range(n_centers)}

        # Time variables for delivery windows
        delivery_time = {(n, m): model.addVar(vtype="C", name=f"DeliveryTime_{n}_{m}") for n in range(n_trucks) for m in range(n_centers)}

        # Additional Variables for Bid Allocation and Drone Optimization
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(n_bids)}
        drone_vars = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(self.n_drones)}
        battery_usage_vars = {(i, d): model.addVar(vtype="I", name=f"BatteryUsage_{i}_{d}") for i in range(n_bids) for d in range(self.n_drones)}
        charging_station_vars = {c: model.addVar(vtype="B", name=f"ChargingStation_{c}") for c in range(self.charging_station_max_count)}

        # Objective: Minimize the total cost including activation, travel time penalty, handling cost, environmental impact, drone usage, and charging station setup
        penalty_per_travel_time = 100
        environmental_cost_factor = 500
        
        objective_expr = quicksum(activation_costs[m] * activate_center[m] for m in range(n_centers)) + \
                         penalty_per_travel_time * quicksum(truck_travel_times[n, m] * allocate_truck[n, m] for n in range(n_trucks) for m in range(n_centers)) + \
                         quicksum(handling_costs[m] * hazard_use[m] for m in range(n_centers)) + \
                         environmental_cost_factor * quicksum(environmental_penalty[m] for m in range(n_centers)) + \
                         quicksum(drone_battery_cost[d] * quicksum(battery_usage_vars[i, d] for i in range(n_bids)) for d in range(self.n_drones)) + \
                         quicksum(charging_station_cost[c] * charging_station_vars[c] for c in range(self.charging_station_max_count))

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

        # Environmental Impact restriction
        for m in range(n_centers):
            model.addCons(environmental_penalty[m] == quicksum(truck_travel_times[n, m] * allocate_truck[n, m] for n in range(n_trucks)) / 100, f"Environmental_Penalty_{m}")

        # Delivery must be within the time window
        for n in range(n_trucks):
            for m in range(n_centers):
                model.addCons(delivery_time[n, m] == time_windows[n], f"Delivery_Time_Window_{n}_{m}")

        # New constraints for bid allocation and drones
        for i, (items, _) in enumerate(bids):
            for item in items:
                for m in range(n_centers):
                    model.addCons(bid_vars[i] <= allocate_truck[item, m], f"Bid_Allocation_{i}_{item}_{m}")

        # Drone assignment and battery usage constraints
        for i in range(n_bids):
            model.addCons(quicksum(drone_vars[d] for d in range(self.n_drones)) <= 1, f"Drone_Assignment_{i}")
            for d in range(self.n_drones):
                model.addCons(battery_usage_vars[i, d] <= drone_max_battery[d] * drone_vars[d], f"Battery_Usage_Limit_{i}_{d}")

        # Charging station capacity constraints
        for c in range(self.charging_station_max_count):
            model.addCons(quicksum(battery_usage_vars[i, d] for i in range(n_bids) for d in range(self.n_drones)) <= charging_station_vars[c] * self.charging_station_capacity, f"Charging_Station_Capacity_{c}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_trucks': 84,
        'n_centers': 148,
        'arrival_rate_interval': (225, 2250),
        'capacity_interval': (506, 1518),
        'activation_cost_interval': (281, 562),
        'ratio': 2835.0,
        'hazard_limit_interval': (37, 187),
        'handling_cost_interval': (186, 759),
        'delivery_window_interval': (405, 1620),
        'n_bids': 100,
        'min_bid_price': 300,
        'max_bid_price': 2000,
        'max_bundle_size': 70,
        'n_drones': 60,
        'drone_battery_interval': (750, 1500),
        'charging_station_max_count': 50,
        'charging_station_cost_interval': (750, 3000),
        'charging_station_capacity': 75,
    }

    urban_logistics_advanced = UrbanLogisticsAdvanced(parameters, seed=seed)
    instance = urban_logistics_advanced.generate_instance()
    solve_status, solve_time = urban_logistics_advanced.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")