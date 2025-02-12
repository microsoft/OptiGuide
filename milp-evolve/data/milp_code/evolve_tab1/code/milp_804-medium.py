import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainOptimization:
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

        # Generating data for the instance
        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_warehouses, self.n_customers))
        capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.n_warehouses)

        return {
            "warehouse_costs": warehouse_costs,
            "delivery_costs": delivery_costs,
            "capacities": capacities,
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        delivery_costs = instance['delivery_costs']
        capacities = instance['capacities']
        
        model = Model("SupplyChainOptimization")
        n_warehouses = len(warehouse_costs)
        n_customers = len(delivery_costs[0])
        
        # Decision variables
        warehouse_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        delivery_vars = {(w, c): model.addVar(vtype="B", name=f"Warehouse_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}

        # Objective: minimize the total cost (warehouse + delivery)
        model.setObjective(quicksum(warehouse_costs[w] * warehouse_vars[w] for w in range(n_warehouses)) +
                           quicksum(delivery_costs[w][c] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)), "minimize")
        
        # Constraints: Each customer is served by exactly one warehouse
        for c in range(n_customers):
            model.addCons(quicksum(delivery_vars[w, c] for w in range(n_warehouses)) == 1, f"Customer_{c}_Assignment")
        
        # Constraints: Only open warehouses can deliver to customers
        for w in range(n_warehouses):
            for c in range(n_customers):
                model.addCons(delivery_vars[w, c] <= warehouse_vars[w], f"Warehouse_{w}_Service_{c}")
        
        # Constraints: Warehouses cannot exceed their capacity
        for w in range(n_warehouses):
            model.addCons(quicksum(delivery_vars[w, c] for c in range(n_customers)) <= capacities[w], f"Warehouse_{w}_Capacity")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 150,
        'n_customers': 112,
        'min_warehouse_cost': 500,
        'max_warehouse_cost': 8000,
        'min_delivery_cost': 75,
        'max_delivery_cost': 2400,
        'min_warehouse_capacity': 5,
        'max_warehouse_capacity': 1875,
    }

    supply_chain_optimizer = SupplyChainOptimization(parameters, seed=42)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")