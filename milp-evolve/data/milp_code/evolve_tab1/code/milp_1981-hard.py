import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class Warehouse:
    """Helper function: Container for a warehouse system."""
    def __init__(self, number_of_warehouses, items, fixed_costs):
        self.number_of_warehouses = number_of_warehouses
        self.items = items
        self.fixed_costs = fixed_costs

class KnapsackOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    # Sample the item weights and values array
    def generate_items(self, number_of_items):
        weights = np.random.randint(1, 20, size=number_of_items)
        values = np.random.randint(10, 100, size=number_of_items)
        return weights, values

    ################# Data Generation #################
    def generate_warehouse(self):
        items = list(range(self.n_items))
        fixed_costs = np.random.randint(50, 100, size=self.n_warehouses)  # Fixed costs for each warehouse
        return Warehouse(self.n_warehouses, items, fixed_costs)

    def generate_instance(self):
        weights, values = self.generate_items(self.n_items)
        warehouse = self.generate_warehouse()
        warehouse_capacities = np.random.normal(150, 30, size=self.n_warehouses).astype(int)
        
        res = {
            'warehouse': warehouse,
            'weights': weights,
            'values': values,
            'warehouse_capacities': warehouse_capacities,
        }
        return res

    ################## PySCIPOpt Modeling #################
    def solve(self, instance):
        warehouse = instance['warehouse']
        weights = instance['weights']
        values = instance['values']
        warehouse_capacities = instance['warehouse_capacities']

        model = Model("KnapsackOptimization")

        # Variables
        item_vars = {(w, i): model.addVar(vtype="B", name=f"Item_{w}_{i}") for w in range(self.n_warehouses) for i in warehouse.items}
        warehouse_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(self.n_warehouses)}  # Binary var indicating warehouse utilization
        
        # Constraints
        for w in range(self.n_warehouses):
            model.addCons(quicksum(item_vars[w, i] * weights[i] for i in warehouse.items) <= warehouse_capacities[w] * warehouse_vars[w], name=f"Capacity_{w}")

        for i in warehouse.items:
            model.addCons(quicksum(item_vars[w, i] for w in range(self.n_warehouses)) <= 1, name=f"ItemAssignment_{i}")
            
        # Ensure each warehouse's fixed cost is only incurred if it is used
        for w in range(self.n_warehouses):
            model.addCons(quicksum(item_vars[w, i] for i in warehouse.items) <= self.big_m * warehouse_vars[w], name=f"WarehouseUsage_{w}")

        # Objective Function
        model.setObjective(
            quicksum(item_vars[w, i] * values[i] for w in range(self.n_warehouses) for i in warehouse.items) 
            - quicksum(warehouse_vars[w] * warehouse.fixed_costs[w] for w in range(self.n_warehouses)), 
            "maximize"
        )
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 50,
        'n_warehouses': 100,
        'big_m': 750,
    }
    
    knapsack_optimization = KnapsackOptimization(parameters, seed=seed)
    instance = knapsack_optimization.generate_instance()
    solve_status, solve_time = knapsack_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")