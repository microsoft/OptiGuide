import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class WarehouseLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        noun_Costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1,
                                       (self.n_customers, self.n_warehouses))
        warehouse_Costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_warehouses)

        return {
            "noun_Costs": noun_Costs,
            "warehouse_Costs": warehouse_Costs
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        noun_Costs = instance['noun_Costs']
        warehouse_Costs = instance['warehouse_Costs']

        model = Model("WarehouseLocation")

        # Decision variables
        warehouse_Open = {j: model.addVar(vtype="B", name=f"WarehouseOpen_{j}") for j in range(self.n_warehouses)}
        noun_Customers = {(i, j): model.addVar(vtype="B", name=f"Customer_{i}_Warehouse_{j}")
                          for i in range(self.n_customers) for j in range(self.n_warehouses)}

        # Objective: minimize the total cost (fixed + transportation costs)
        objective_expr = quicksum(warehouse_Costs[j] * warehouse_Open[j] for j in range(self.n_warehouses)) + \
                         quicksum(noun_Costs[i, j] * noun_Customers[i, j] for i in range(self.n_customers)
                                                                                  for j in range(self.n_warehouses))
        model.setObjective(objective_expr, "minimize")

        # Constraints: Each customer is served by exactly one warehouse
        for i in range(self.n_customers):
            model.addCons(quicksum(noun_Customers[i, j] for j in range(self.n_warehouses)) == 1, f"Customer_{i}")

        # Constraints: A customer is served by a warehouse only if the warehouse is open
        for i in range(self.n_customers):
            for j in range(self.n_warehouses):
                model.addCons(noun_Customers[i, j] <= warehouse_Open[j], f"Serve_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 600,
        'n_warehouses': 25,
        'min_transport_cost': 25,
        'max_transport_cost': 60,
        'min_fixed_cost': 300,
        'max_fixed_cost': 1000,
    }

    warehouse_location = WarehouseLocation(parameters, seed=42)
    instance = warehouse_location.generate_instance()
    solve_status, solve_time = warehouse_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")