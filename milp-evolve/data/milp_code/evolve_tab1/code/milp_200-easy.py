import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class AdvancedCombinatorialAuctionWithFLPAndDroneOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data generation #################
    def generate_instance(self):
        assert self.min_value >= 0 and self.max_value >= self.min_value
        assert self.add_item_prob >= 0 and self.add_item_prob <= 1

        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)
        bids = []

        while len(bids) < self.n_bids:
            bundle_size = np.random.randint(1, self.max_bundle_size + 1)
            bundle = np.random.choice(self.n_items, size=bundle_size, replace=False)
            price = values[bundle].sum()

            if price < 0:
                continue

            bids.append((bundle.tolist(), price))

        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        # Facility data generation
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=len(bids)).tolist()
        transaction_costs = np.random.uniform(0.1, 0.5, size=len(bids)).tolist()
        capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()

        # Generate mutual exclusivity groups as logical conditions
        bid_requirements = {}
        for bid in range(len(bids)):
            required_facility = random.randint(0, n_facilities - 1)
            bid_requirements[bid] = required_facility

        # Drone data generation
        drone_max_battery = np.random.randint(50, 100, size=self.n_drones).tolist()  # battery capacity in arbitrary units
        drone_battery_cost = np.random.gamma(shape=2.0, scale=1.0, size=self.n_drones).tolist()  # cost per battery usage unit
        charging_station_cost = np.random.uniform(50, 200, size=self.charging_station_max_count).tolist()  # cost of setting up each charging station

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "transaction_costs": transaction_costs,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "bid_requirements": bid_requirements,
            "drone_max_battery": drone_max_battery,
            "drone_battery_cost": drone_battery_cost,
            "charging_station_cost": charging_station_cost
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        transaction_costs = instance['transaction_costs']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        bid_requirements = instance['bid_requirements']
        drone_max_battery = instance['drone_max_battery']
        drone_battery_cost = instance['drone_battery_cost']
        charging_station_cost = instance['charging_station_cost']

        model = Model("AdvancedCombinatorialAuctionWithFLPAndDroneOptimization")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(len(bids)) for j in range(n_facilities)}
        
        item_vars = {i: model.addVar(vtype="I", name=f"ItemWon_{i}") for i in range(self.n_items)}

        ## New drone-related variables
        drone_vars = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(self.n_drones)}
        battery_usage_vars = {(i, d): model.addVar(vtype="I", name=f"BatteryUsage_{i}_{d}") for i in range(len(bids)) for d in range(self.n_drones)}
        charging_station_vars = {c: model.addVar(vtype="B", name=f"ChargingStation_{c}") for c in range(self.charging_station_max_count)}

        ## Objective Function with new terms
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * quicksum(x_vars[i, j] for j in range(n_facilities)) for i in range(len(bids))) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(transaction_costs[i] * bid_vars[i] for i in range(len(bids))) \
                         - quicksum(drone_battery_cost[d] * quicksum(battery_usage_vars[i,d] for i in range(len(bids))) for d in range(self.n_drones)) \
                         - quicksum(charging_station_cost[c] * charging_station_vars[c] for c in range(self.charging_station_max_count))

        ## Constraints: Each item can only be part of one accepted bid
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        ## Bid assignment to facility
        for i in range(len(bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i], f"BidFacility_{i}")

        ## Facility capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(bids))) <= capacity[j] * y_vars[j], f"FacilityCapacity_{j}")
        
        ## Logical constraints: enforce bid-facility dependency
        for bid, facility in bid_requirements.items():
            model.addCons(bid_vars[bid] <= y_vars[facility], f"LogicalDependency_Bid_{bid}_Facility_{facility}")

        ## Linking ItemWon variables to bids
        for item in range(self.n_items):
            model.addCons(item_vars[item] == quicksum(bid_vars[bid_idx] for bid_idx in bids_per_item[item]), f"LinkItem_{item}")

        ## Drone assignment to bids and battery usage
        for i in range(len(bids)):
            model.addCons(quicksum(drone_vars[d] for d in range(self.n_drones)) <= 1, f"DroneAssignment_{i}")
            for d in range(self.n_drones):
                model.addCons(battery_usage_vars[i,d] <= drone_max_battery[d] * drone_vars[d], f"BatteryLimit_{i}_{d}")

        ## Charging station setup constraints
        for c in range(self.charging_station_max_count):
            model.addCons(quicksum(battery_usage_vars[i,d] for i in range(len(bids)) for d in range(self.n_drones)) <= charging_station_vars[c] * self.charging_station_capacity, f"ChargingStationCapacity_{c}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 2500,
        'n_bids': 75,
        'min_value': 50,
        'max_value': 5000,
        'max_bundle_size': 52,
        'add_item_prob': 0.24,
        'facility_min_count': 180,
        'facility_max_count': 750,
        'bidder_min_items': 3,
        'bidder_max_items': 50,
        'n_drones': 50,
        'charging_station_max_count': 20,
        'charging_station_capacity': 100
    }

    auction = AdvancedCombinatorialAuctionWithFLPAndDroneOptimization(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")