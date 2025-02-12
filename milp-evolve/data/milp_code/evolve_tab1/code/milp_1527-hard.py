import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class WarehouseLocationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_warehouses > 0 and self.n_stores > 0
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_warehouses, self.n_stores))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_warehouses)
        store_demand = np.random.randint(1, 10, self.n_stores)
        operational_limits = np.random.uniform(self.min_operational_limit, self.max_operational_limit, self.n_warehouses)
        distances = np.random.uniform(0, self.max_distance, (self.n_warehouses, self.n_stores))
        service_threshold = np.random.randint(1, self.n_stores // 2, self.n_warehouses)
        
        return {
            "warehouse_costs": warehouse_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "store_demand": store_demand,
            "operational_limits": operational_limits,
            "distances": distances,
            "service_threshold": service_threshold,
        }

    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        store_demand = instance['store_demand']
        operational_limits = instance['operational_limits']
        distances = instance['distances']
        service_threshold = instance['service_threshold']

        model = Model("WarehouseLocationOptimization")
        n_warehouses = len(warehouse_costs)
        n_stores = len(transport_costs[0])

        # Decision variables
        open_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        supply_vars = {(w, s): model.addVar(vtype="C", name=f"Supply_{w}_{s}") for w in range(n_warehouses) for s in range(n_stores)}
        serve_vars = {(w, s): model.addVar(vtype="B", name=f"Serve_{w}_{s}") for w in range(n_warehouses) for s in range(n_stores)}

        # Objective: minimize the total cost including warehouse operating costs and transport costs.
        model.setObjective(
            quicksum(warehouse_costs[w] * open_vars[w] for w in range(n_warehouses)) +
            quicksum(transport_costs[w, s] * supply_vars[w, s] for w in range(n_warehouses) for s in range(n_stores)),
            "minimize"
        )

        # Constraints: Each store's demand is met by the warehouses
        for s in range(n_stores):
            model.addCons(quicksum(supply_vars[w, s] for w in range(n_warehouses)) == store_demand[s], f"Store_{s}_Demand")

        # Constraints: Only open warehouses can supply stores
        for w in range(n_warehouses):
            for s in range(n_stores):
                model.addCons(supply_vars[w, s] <= operational_limits[w] * open_vars[w], f"Warehouse_{w}_Supply_{s}")

        # Constraints: Warehouses cannot exceed their capacities
        for w in range(n_warehouses):
            model.addCons(quicksum(supply_vars[w, s] for s in range(n_stores)) <= capacities[w], f"Warehouse_{w}_Capacity")

        # Logical constraints with service
        for s in range(n_stores):
            for w in range(n_warehouses):
                model.addCons(serve_vars[w, s] <= open_vars[w], f"Serve_Open_Constraint_{w}_{s}")
                model.addCons(distances[w, s] * open_vars[w] <= self.max_distance * serve_vars[w, s], f"Distance_Constraint_{w}_{s}")
            model.addCons(quicksum(serve_vars[w, s] for w in range(n_warehouses)) >= 1, f"Store_{s}_Service")

        # New constraint: Ensure each opened warehouse serves at least the minimum service threshold
        for w in range(n_warehouses):
            model.addCons(quicksum(serve_vars[w, s] for s in range(n_stores)) >= service_threshold[w] * open_vars[w], f"Warehouse_{w}_Service_Threshold")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 73
    parameters = {
        'n_warehouses': 100,
        'n_stores': 100,
        'min_transport_cost': 20,
        'max_transport_cost': 900,
        'min_warehouse_cost': 1600,
        'max_warehouse_cost': 3000,
        'min_capacity': 300,
        'max_capacity': 1400,
        'min_operational_limit': 200,
        'max_operational_limit': 2000,
        'max_distance': 1000,
    }

    warehouse_optimizer = WarehouseLocationOptimization(parameters, seed=seed)
    instance = warehouse_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")