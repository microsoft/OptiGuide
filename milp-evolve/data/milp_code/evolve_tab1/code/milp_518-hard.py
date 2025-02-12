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

    ################# Data Generation #################
    def generate_instance(self):
        n_warehouses = random.randint(self.min_warehouses, self.max_warehouses)
        n_locations = random.randint(self.min_locations, self.max_locations)

        # Cost matrices and Big M
        service_costs = np.random.randint(10, 100, size=(n_warehouses, n_locations))
        fixed_costs = np.random.randint(500, 1500, size=n_warehouses)
        big_m = 1000  # Arbitrary large number for Big M formulation

        res = {
            'n_warehouses': n_warehouses,
            'n_locations': n_locations,
            'service_costs': service_costs,
            'fixed_costs': fixed_costs,
            'big_m': big_m
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_warehouses = instance['n_warehouses']
        n_locations = instance['n_locations']
        service_costs = instance['service_costs']
        fixed_costs = instance['fixed_costs']
        big_m = instance['big_m']

        model = Model("WarehouseLocationOptimization")

        # Variables
        y = {i: model.addVar(vtype="B", name=f"y_{i}") for i in range(n_warehouses)}
        x = {}
        for i in range(n_warehouses):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Objective function: Minimize total cost
        total_cost = quicksum(x[i, j] * service_costs[i, j] for i in range(n_warehouses) for j in range(n_locations)) + \
                     quicksum(y[i] * fixed_costs[i] for i in range(n_warehouses))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] for i in range(n_warehouses)) == 1, name=f"location_coverage_{j}")

        # Big-M Formulation for logical constraints: A warehouse can only serve locations if it is open
        for i in range(n_warehouses):
            for j in range(n_locations):
                model.addCons(x[i, j] - y[i] <= 0, name=f"big_m_warehouse_to_location_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_warehouses': 60,
        'max_warehouses': 75,
        'min_locations': 100,
        'max_locations': 150,
    }

    optimization = WarehouseLocationOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")