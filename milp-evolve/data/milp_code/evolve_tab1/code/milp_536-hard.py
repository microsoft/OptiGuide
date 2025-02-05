import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ProductionDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_factories = random.randint(self.min_factories, self.max_factories)
        n_locations = random.randint(self.min_locations, self.max_locations)

        # Cost and time matrices
        production_costs = np.random.randint(1000, 5000, size=n_factories)
        delivery_costs = np.random.randint(50, 200, size=(n_factories, n_locations))
        production_time = np.random.randint(1, 10, size=n_factories)
        inventory_capacity = np.random.randint(50, 200, size=n_factories)
        delivery_time_windows = np.random.randint(1, 10, size=n_locations)

        res = {
            'n_factories': n_factories,
            'n_locations': n_locations,
            'production_costs': production_costs,
            'delivery_costs': delivery_costs,
            'production_time': production_time,
            'inventory_capacity': inventory_capacity,
            'delivery_time_windows': delivery_time_windows
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_factories = instance['n_factories']
        n_locations = instance['n_locations']
        production_costs = instance['production_costs']
        delivery_costs = instance['delivery_costs']
        production_time = instance['production_time']
        inventory_capacity = instance['inventory_capacity']
        delivery_time_windows = instance['delivery_time_windows']

        model = Model("ProductionDeliveryOptimization")

        # Variables
        y = {i: model.addVar(vtype="B", name=f"y_{i}") for i in range(n_factories)}  # Whether a factory is open
        x = {}
        for i in range(n_factories):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")  # Delivery from factory i to location j
        
        production_status = {i: model.addVar(vtype="B", name=f"production_status_{i}") for i in range(n_factories)}

        # Objective function: Minimize total production and delivery costs
        total_cost = quicksum(production_status[i] * production_costs[i] for i in range(n_factories)) + \
                     quicksum(x[i, j] * delivery_costs[i, j] for i in range(n_factories) for j in range(n_locations))

        model.setObjective(total_cost, "minimize")

        # Constraints
        # Ensure each location is served by exactly one factory
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] for i in range(n_factories)) == 1, name=f"location_coverage_{j}")

        # Logical constraints: A factory can deliver products only if it's producing
        for i in range(n_factories):
            for j in range(n_locations):
                model.addCons(x[i, j] <= production_status[i], name=f"factory_to_location_{i}_{j}")

        # Ensure production constraints: a factory can produce within its capacity
        for i in range(n_factories):
            model.addCons(production_status[i] * inventory_capacity[i] >= quicksum(x[i, j] for j in range(n_locations)), name=f"production_capacity_{i}")

        # Ensure delivery time window constraints
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * production_time[i] for i in range(n_factories)) <= delivery_time_windows[j], name=f"delivery_time_window_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_factories': 10,
        'max_factories': 50,
        'min_locations': 40,
        'max_locations': 2000,
    }

    optimization = ProductionDeliveryOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")