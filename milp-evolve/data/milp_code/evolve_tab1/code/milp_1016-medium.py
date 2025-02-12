import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexManufacturingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_warehouses > 0 and self.n_customers > 0
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_delivery_cost >= 0 and self.max_delivery_cost >= self.min_delivery_cost
        assert self.min_warehouse_capacity > 0 and self.max_warehouse_capacity >= self.min_warehouse_capacity
        
        # Existing data generation
        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_warehouses, self.n_customers))
        capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.n_warehouses)
        delivery_time_windows = np.random.randint(0, 24, (self.n_customers, 2))
        delivery_time_windows[:, 1] = np.clip(delivery_time_windows[:, 1], a_min=delivery_time_windows[:, 0] + 1, a_max=24)
        traffic_penalties = np.random.randint(1, 5, (self.n_warehouses, self.n_customers))
        max_emissions = 5000
        emissions_per_km = np.random.uniform(0.1, 1.0, (self.n_warehouses, self.n_customers))
        weights = np.random.randint(1, 10, (self.n_warehouses, self.n_customers))
        volumes = np.random.randint(1, 5, (self.n_warehouses, self.n_customers))
        vehicle_weight_capacity = np.random.randint(self.min_vehicle_weight_capacity, self.max_vehicle_weight_capacity + 1, self.n_warehouses)
        vehicle_volume_capacity = np.random.randint(self.min_vehicle_volume_capacity, self.max_vehicle_volume_capacity + 1, self.n_warehouses)
        g = nx.barabasi_albert_graph(self.n_customers, 5, seed=self.seed)

        # New data generation for bids and budget
        machine_maintenance = np.random.randint(0, 2, (self.n_warehouses, 24))  # Binary maintenance schedule over 24 periods
        production_capacity_per_time = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, (self.n_warehouses, 24))
        raw_material_inventory = np.random.randint(50, 200, (self.n_warehouses, 24))  # Initial raw material inventory for each warehouse over each time period
        production_cost_per_time = np.random.randint(self.min_production_cost, self.max_production_cost + 1, (self.n_warehouses, 24))
        unmet_demand_penalty = 1000  # Arbitrary penalty cost for unmet demand
        machine_downtime_cost = 500  # Arbitrary cost associated with machine downtime

        bids = []
        for _ in range(self.n_bids):
            private_interests = np.random.rand(self.n_items)
            private_values = np.random.uniform(self.min_value, self.max_value, self.n_items) * (2 * private_interests - 1)
            initial_item = np.random.choice(self.n_items, p=private_interests / private_interests.sum())
            bundle_mask = np.zeros(self.n_items, dtype=bool)
            bundle_mask[initial_item] = True
            while np.random.rand() < self.add_item_prob and bundle_mask.sum() < self.n_items:
                next_item = np.random.choice(self.n_items, p=(private_interests * ~bundle_mask) / (private_interests * ~bundle_mask).sum())
                bundle_mask[next_item] = True
            bundle = np.nonzero(bundle_mask)[0]
            price = private_values[bundle].sum() + len(bundle) ** 1.25 + np.random.exponential(scale=100)
            if price > 0:
                bids.append((list(bundle), price))
        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)
        budget = np.random.uniform(self.min_budget, self.max_budget)

        return {
            "warehouse_costs": warehouse_costs,
            "delivery_costs": delivery_costs,
            "capacities": capacities,
            "graph_deliveries": nx.to_numpy_array(g),
            "time_windows": delivery_time_windows,
            "traffic_penalties": traffic_penalties,
            "max_emissions": max_emissions,
            "emissions_per_km": emissions_per_km,
            "weights": weights,
            "volumes": volumes,
            "vehicle_weight_capacity": vehicle_weight_capacity,
            "vehicle_volume_capacity": vehicle_volume_capacity,
            "machine_maintenance": machine_maintenance,
            "production_capacity_per_time": production_capacity_per_time,
            "raw_material_inventory": raw_material_inventory,
            "production_cost_per_time": production_cost_per_time,
            "unmet_demand_penalty": unmet_demand_penalty,
            "machine_downtime_cost": machine_downtime_cost,
            "bids": bids,
            "bids_per_item": bids_per_item,
            "budget": budget
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        delivery_costs = instance['delivery_costs']
        capacities = instance['capacities']
        graph_deliveries = instance['graph_deliveries']
        time_windows = instance['time_windows']
        traffic_penalties = instance['traffic_penalties']
        max_emissions = instance['max_emissions']
        emissions_per_km = instance['emissions_per_km']
        weights = instance['weights']
        volumes = instance['volumes']
        vehicle_weight_capacity = instance['vehicle_weight_capacity']
        vehicle_volume_capacity = instance['vehicle_volume_capacity']
        machine_maintenance = instance['machine_maintenance']
        production_capacity_per_time = instance['production_capacity_per_time']
        raw_material_inventory = instance['raw_material_inventory']
        production_cost_per_time = instance['production_cost_per_time']
        unmet_demand_penalty = instance['unmet_demand_penalty']
        machine_downtime_cost = instance['machine_downtime_cost']
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        budget = instance['budget']
        
        model = Model("ComplexManufacturingOptimization")
        n_warehouses = len(warehouse_costs)
        n_customers = len(delivery_costs[0])
        T = machine_maintenance.shape[1]  # Number of time periods

        # Decision variables
        warehouse_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        delivery_vars = {(w, c): model.addVar(vtype="I", lb=0, ub=10, name=f"Warehouse_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}
        delivery_times = {(w, c): model.addVar(vtype="I", lb=0, ub=23, name=f"DeliveryTime_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}
        production_vars = {(w, t): model.addVar(vtype="I", lb=0, name=f"Production_{w}_{t}") for w in range(n_warehouses) for t in range(T)}
        inventory_vars = {(w, t): model.addVar(vtype="I", lb=0, name=f"Inventory_{w}_{t}") for w in range(n_warehouses) for t in range(T)}
        unmet_demand_vars = {c: model.addVar(vtype="I", lb=0, name=f"UnmetDemand_{c}") for c in range(n_customers)}
        downtime_vars = {(w, t): model.addVar(vtype="I", name=f"Downtime_{w}_{t}") for w in range(n_warehouses) for t in range(T)}
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        expenditure_vars = {i: model.addVar(vtype="C", name=f"Expenditure_{i}") for i in range(len(bids))}

        # Objective: minimize the total cost (warehouse + delivery + traffic penalties + emissions + production + penalties)
        total_cost = quicksum(warehouse_costs[w] * warehouse_vars[w] for w in range(n_warehouses)) + \
                     quicksum((delivery_costs[w][c] + traffic_penalties[w][c]) * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) + \
                     quicksum(emissions_per_km[w][c] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) + \
                     quicksum(production_cost_per_time[w, t] * production_vars[w, t] for w in range(n_warehouses) for t in range(T)) + \
                     quicksum(unmet_demand_penalty * unmet_demand_vars[c] for c in range(n_customers)) + \
                     quicksum(machine_downtime_cost * downtime_vars[w, t] for w in range(n_warehouses) for t in range(T)) + \
                     quicksum(expenditure_vars[i] for i in range(len(bids))) / budget

        model.setObjective(total_cost, "minimize")

        # Constraints: Each customer must be served, but not exceeding social network capacity
        for c in range(n_customers):
            model.addCons(quicksum(delivery_vars[w, c] for w in range(n_warehouses)) == quicksum(graph_deliveries[c]), f"CustomerDemand_{c}")
        
        # Constraints: Only open warehouses can deliver to customers
        for w in range(n_warehouses):
            for c in range(n_customers):
                model.addCons(delivery_vars[w, c] <= 10 * warehouse_vars[w], f"Warehouse_{w}_Service_{c}")
  
        # Constraints: Warehouses cannot exceed their capacity
        for w in range(n_warehouses):
            model.addCons(quicksum(delivery_vars[w, c] for c in range(n_customers)) <= capacities[w], f"Warehouse_{w}_Capacity")

        # Constraints: Deliveries must occur within time windows
        for c in range(n_customers):
            for w in range(n_warehouses):
                model.addCons(delivery_times[w, c] >= time_windows[c, 0], f"TimeWindowStart_{w}_{c}")
                model.addCons(delivery_times[w, c] <= time_windows[c, 1], f"TimeWindowEnd_{w}_{c}")

        # Constraints: Limit on carbon emissions
        total_emissions = quicksum(emissions_per_km[w][c] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers))
        model.addCons(total_emissions <= max_emissions, "MaxEmissions")

        # Constraints: Weight and Volume capacity constraints
        for w in range(n_warehouses):
            model.addCons(quicksum(weights[w, c] * delivery_vars[w, c] for c in range(n_customers)) <= vehicle_weight_capacity[w], f"WeightCapacity_{w}")
            model.addCons(quicksum(volumes[w, c] * delivery_vars[w, c] for c in range(n_customers)) <= vehicle_volume_capacity[w], f"VolumeCapacity_{w}")
        
        # Constraints for production capacity, raw material, and maintenance
        for w in range(n_warehouses):
            for t in range(T):
                model.addCons(production_vars[w, t] <= production_capacity_per_time[w, t], f"ProductionCapacity_{w}_{t}")
                model.addCons(inventory_vars[w, t] >= production_vars[w, t], f"InventoryCheck_{w}_{t}")
                model.addCons(downtime_vars[w, t] == machine_maintenance[w, t], f"Downtime_{w}_{t}")
        
                if t > 0:
                    model.addCons(inventory_vars[w, t] <= inventory_vars[w, t-1] - production_vars[w, t-1], f"InventoryFlow_{w}_{t}")

        # Constraints for bidding and expenditure
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")
        
        model.addCons(quicksum(expenditure_vars[i] for i in range(len(bids))) <= budget, "Budget")
        
        for i, (bundle, price) in enumerate(bids):
            model.addCons(expenditure_vars[i] == price * bid_vars[i], f"Expenditure_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 666,
        'n_customers': 21,
        'min_warehouse_cost': 2500,
        'max_warehouse_cost': 8000,
        'min_delivery_cost': 567,
        'max_delivery_cost': 675,
        'min_warehouse_capacity': 472,
        'max_warehouse_capacity': 527,
        'min_vehicle_weight_capacity': 1500,
        'max_vehicle_weight_capacity': 3000,
        'min_vehicle_volume_capacity': 224,
        'max_vehicle_volume_capacity': 3000,
        'min_production_cost': 375,
        'max_production_cost': 375,
        'n_items': 1000,
        'n_bids': 1350,
        'min_value': 50,
        'max_value': 500,
        'add_item_prob': 0.52,
        'min_budget': 10000,
        'max_budget': 50000,
    }
    
    complex_manufacturing_optimizer = ComplexManufacturingOptimization(parameters, seed=42)
    instance = complex_manufacturing_optimizer.generate_instance()
    solve_status, solve_time, objective_value = complex_manufacturing_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")