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

        # New data generation
        machine_maintenance = np.random.randint(0, 2, (self.n_warehouses, 24))  # Binary maintenance schedule over 24 periods
        production_capacity_per_time = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, (self.n_warehouses, 24))
        raw_material_inventory = np.random.randint(50, 200, (self.n_warehouses, 24))  # Initial raw material inventory for each warehouse over each time period
        production_cost_per_time = np.random.randint(self.min_production_cost, self.max_production_cost + 1, (self.n_warehouses, 24))  # Cost per production unit per time period
        unmet_demand_penalty = 1000  # Arbitrary penalty cost for unmet demand
        machine_downtime_cost = 500  # Arbitrary cost associated with machine downtime

        # New data generation for added complexity
        market_price_fluctuations = np.random.uniform(self.min_market_price_fluctuation, self.max_market_price_fluctuation, self.n_warehouses)
        recycled_materials_availability = np.random.randint(0, 2, (self.n_warehouses, 24))
        regulatory_compliance_costs = np.random.uniform(self.min_regulatory_cost, self.max_regulatory_cost, (self.n_warehouses, 24))
        customer_satisfaction_tiers = np.random.randint(0, 3, self.n_customers)  # 0: Low, 1: Medium, 2: High
        customer_satisfaction_penalties = [100, 50, 0]  # Penalties corresponding to satisfaction tiers

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
            "market_price_fluctuations": market_price_fluctuations,
            "recycled_materials_availability": recycled_materials_availability,
            "regulatory_compliance_costs": regulatory_compliance_costs,
            "customer_satisfaction_tiers": customer_satisfaction_tiers,
            "customer_satisfaction_penalties": customer_satisfaction_penalties
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
        market_price_fluctuations = instance['market_price_fluctuations']
        recycled_materials_availability = instance['recycled_materials_availability']
        regulatory_compliance_costs = instance['regulatory_compliance_costs']
        customer_satisfaction_tiers = instance['customer_satisfaction_tiers']
        customer_satisfaction_penalties = instance['customer_satisfaction_penalties']
        
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
        recycled_material_vars = {(w, t): model.addVar(vtype="B", name=f"RecycledMaterial_{w}_{t}") for w in range(n_warehouses) for t in range(T)}
        market_price_vars = {(w, t): model.addVar(vtype="C", name=f"MarketPrice_{w}_{t}") for w in range(n_warehouses) for t in range(T)}

        # Objective: minimize the total cost (warehouse + delivery + traffic penalties + emissions + production + penalties)
        total_cost = quicksum(warehouse_costs[w] * warehouse_vars[w] for w in range(n_warehouses)) + \
                     quicksum((delivery_costs[w][c] + traffic_penalties[w][c]) * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) + \
                     quicksum(emissions_per_km[w][c] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) + \
                     quicksum(production_cost_per_time[w, t] * production_vars[w, t] for w in range(n_warehouses) for t in range(T)) + \
                     quicksum(unmet_demand_penalty * unmet_demand_vars[c] for c in range(n_customers)) + \
                     quicksum(machine_downtime_cost * downtime_vars[w, t] for w in range(n_warehouses) for t in range(T)) + \
                     quicksum(regulatory_compliance_costs[w, t] * (1 - recycled_material_vars[w, t]) for w in range(n_warehouses) for t in range(T)) + \
                     quicksum(customer_satisfaction_penalties[customer_satisfaction_tiers[c]] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) + \
                     quicksum(market_price_fluctuations[w] * market_price_vars[w, t] for w in range(n_warehouses) for t in range(T))

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

        # Constraints for recycled materials and market prices
        for w in range(n_warehouses):
            for t in range(T):
                model.addCons(production_vars[w, t] >= recycled_material_vars[w, t] * raw_material_inventory[w, t], f"RecycledMaterialUsage_{w}_{t}")
                model.addCons(market_price_vars[w, t] <= production_cost_per_time[w, t] * (1 + market_price_fluctuations[w]), f"MarketPriceFluctuation_{w}_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 222,
        'n_customers': 42,
        'min_warehouse_cost': 1250,
        'max_warehouse_cost': 8000,
        'min_delivery_cost': 1134,
        'max_delivery_cost': 1350,
        'min_warehouse_capacity': 630,
        'max_warehouse_capacity': 1054,
        'min_vehicle_weight_capacity': 1500,
        'max_vehicle_weight_capacity': 3000,
        'min_vehicle_volume_capacity': 112,
        'max_vehicle_volume_capacity': 600,
        'min_production_cost': 500,
        'max_production_cost': 750,
        'min_market_price_fluctuation': 0.05,
        'max_market_price_fluctuation': 0.2,
        'min_regulatory_cost': 100,
        'max_regulatory_cost': 500,
    }

    complex_manufacturing_optimizer = ComplexManufacturingOptimization(parameters, seed=42)
    instance = complex_manufacturing_optimizer.generate_instance()
    solve_status, solve_time, objective_value = complex_manufacturing_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")