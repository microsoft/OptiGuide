import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class Warehouse:
    """Helper function: Container for a warehouse system."""
    def __init__(self, number_of_warehouses, items):
        self.number_of_warehouses = number_of_warehouses
        self.items = items

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
        return Warehouse(self.n_warehouses, items)

    def generate_instance(self):
        weights, values = self.generate_items(self.n_items)
        warehouse = self.generate_warehouse()
        warehouse_capacities = np.random.randint(50, 200, size=self.n_warehouses)
        
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

        # Constraints
        for w in range(self.n_warehouses):
            model.addCons(quicksum(item_vars[w, i] * weights[i] for i in warehouse.items) <= warehouse_capacities[w], name=f"Capacity_{w}")

        for i in warehouse.items:
            model.addCons(quicksum(item_vars[w, i] for w in range(self.n_warehouses)) <= 1, name=f"ItemAssignment_{i}")

        # Objective Function
        model.setObjective(
            quicksum(item_vars[w, i] * values[i] for w in range(self.n_warehouses) for i in warehouse.items), 
            "maximize"
        )
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 1000,
        'n_warehouses': 20,
    }
    
    knapsack_optimization = KnapsackOptimization(parameters, seed=seed)
    instance = knapsack_optimization.generate_instance()
    solve_status, solve_time = knapsack_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")