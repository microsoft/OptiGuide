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
        
        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_warehouses, self.n_customers))
        capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.n_warehouses)
        
        # Additional data for delivery time windows
        delivery_time_windows = np.random.randint(self.min_time_window, self.max_time_window + 1, (self.n_warehouses, self.n_customers))
        
        # Enhanced dataset for overtime delivery costs
        overtime_delivery_costs = np.random.normal(loc=50, scale=10, size=(self.n_warehouses, self.n_customers))

        return {
            "warehouse_costs": warehouse_costs,
            "delivery_costs": delivery_costs,
            "capacities": capacities,
            "delivery_time_windows": delivery_time_windows,
            "overtime_delivery_costs": overtime_delivery_costs
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        delivery_costs = instance['delivery_costs']
        capacities = instance['capacities']
        delivery_time_windows = instance['delivery_time_windows']
        overtime_delivery_costs = instance['overtime_delivery_costs']
        
        model = Model("SupplyChainOptimization")
        n_warehouses = len(warehouse_costs)
        n_customers = len(delivery_costs[0])
        
        # Decision variables
        warehouse_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        delivery_vars = {(w, c): model.addVar(vtype="B", name=f"Warehouse_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}
        overflow = model.addVar(vtype="C", lb=0, name="Overflow")
        
        # Delivery cost variables
        delivery_cost_vars = {(w, c): model.addVar(vtype="C", lb=0, ub=self.max_delivery_cost, name=f"DelCost_{w}_Customer_{c}") 
                              for w in range(n_warehouses) for c in range(n_customers)}

        # Objective: minimize the total cost (warehouse + delivery + overflow penalties)
        model.setObjective(quicksum(warehouse_costs[w] * warehouse_vars[w] for w in range(n_warehouses)) +
                           quicksum(delivery_cost_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) +
                           1000 * overflow, "minimize")
        
        # Constraints: Each customer is served by exactly one warehouse
        for c in range(n_customers):
            model.addCons(quicksum(delivery_vars[w, c] for w in range(n_warehouses)) == 1, f"Customer_{c}_Assignment")
        
        # Constraints: Only open warehouses can deliver to customers
        for w in range(n_warehouses):
            for c in range(n_customers):
                model.addCons(delivery_vars[w, c] <= warehouse_vars[w], f"Warehouse_{w}_Service_{c}")
        
        # Constraints: Warehouses cannot exceed their capacity
        for w in range(n_warehouses):
            model.addCons(quicksum(delivery_vars[w, c] for c in range(n_customers)) <= capacities[w] + overflow, f"Warehouse_{w}_Capacity")

        # Constraints: Ensure delivery time windows with convex hull formulation
        for w in range(n_warehouses):
            for c in range(n_customers):
                model.addCons(delivery_cost_vars[w, c] >= delivery_time_windows[w][c] * delivery_vars[w, c], f"DelTimeWin_{w}_Customer_{c}")
                model.addCons(delivery_cost_vars[w, c] <= self.max_delivery_cost * delivery_vars[w, c], f"DelTimeUpper_{w}_Customer_{c}")
                # Incorporate overtime costs
                model.addCons(delivery_cost_vars[w, c] >= overtime_delivery_costs[w][c] * delivery_vars[w, c], f"OvertimeCost_{w}_Customer_{c}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 18,
        'n_customers': 1000,
        'min_warehouse_cost': 3000,
        'max_warehouse_cost': 10000,
        'min_delivery_cost': 9,
        'max_delivery_cost': 500,
        'min_warehouse_capacity': 126,
        'max_warehouse_capacity': 3000,
        'min_time_window': 45,
        'max_time_window': 562,
    }

    supply_chain_optimizer = SupplyChainOptimization(parameters, seed=42)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")