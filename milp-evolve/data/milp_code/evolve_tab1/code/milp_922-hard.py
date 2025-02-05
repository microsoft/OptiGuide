import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexLogisticsOptimization:
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
        
        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_warehouses, self.n_customers))
        capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.n_warehouses)
        
        # For time windows assuming hours of the day (24 hours)
        delivery_time_windows = np.random.randint(0, 24, (self.n_customers, 2))
        delivery_time_windows[:, 1] = np.clip(delivery_time_windows[:, 1], a_min=delivery_time_windows[:, 0] + 1, a_max=24)
        
        traffic_penalties = np.random.randint(1, 5, (self.n_warehouses, self.n_customers))
        
        max_emissions = 5000  # Arbitrary upper limit for demonstration
        emissions_per_km = np.random.uniform(0.1, 1.0, (self.n_warehouses, self.n_customers))
        
        weights = np.random.randint(1, 10, (self.n_warehouses, self.n_customers))
        volumes = np.random.randint(1, 5, (self.n_warehouses, self.n_customers))
        vehicle_weight_capacity = np.random.randint(self.min_vehicle_weight_capacity, self.max_vehicle_weight_capacity + 1, self.n_warehouses)
        vehicle_volume_capacity = np.random.randint(self.min_vehicle_volume_capacity, self.max_vehicle_volume_capacity + 1, self.n_warehouses)

        g = nx.barabasi_albert_graph(self.n_customers, 5, seed=self.seed)

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
            "vehicle_volume_capacity": vehicle_volume_capacity
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
        
        model = Model("ComplexLogisticsOptimization")
        n_warehouses = len(warehouse_costs)
        n_customers = len(delivery_costs[0])
        
        # Decision variables
        warehouse_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        delivery_vars = {(w, c): model.addVar(vtype="I", lb=0, ub=10, name=f"Warehouse_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}
        delivery_times = {(w, c): model.addVar(vtype="I", lb=0, ub=23, name=f"DeliveryTime_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}

        # Objective: minimize the total cost (warehouse + delivery + traffic penalties + emissions)
        total_cost = quicksum(warehouse_costs[w] * warehouse_vars[w] for w in range(n_warehouses)) + \
                     quicksum((delivery_costs[w][c] + traffic_penalties[w][c]) * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) + \
                     quicksum(emissions_per_km[w][c] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers))
        
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

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 222,
        'n_customers': 84,
        'min_warehouse_cost': 1250,
        'max_warehouse_cost': 8000,
        'min_delivery_cost': 378,
        'max_delivery_cost': 1350,
        'min_warehouse_capacity': 630,
        'max_warehouse_capacity': 2109,
        'min_vehicle_weight_capacity': 750,
        'max_vehicle_weight_capacity': 1500,
        'min_vehicle_volume_capacity': 150,
        'max_vehicle_volume_capacity': 200,
    }

    complex_logistics_optimizer = ComplexLogisticsOptimization(parameters, seed=42)
    instance = complex_logistics_optimizer.generate_instance()
    solve_status, solve_time, objective_value = complex_logistics_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")