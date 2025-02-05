import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class MediaContentDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_contents = random.randint(self.min_contents, self.max_contents)
        n_spots = random.randint(self.min_spots, self.max_spots)
        n_storage_units = random.randint(self.min_storage_units, self.max_storage_units)

        # Cost matrices
        content_costs = np.random.randint(15, 125, size=(n_contents, n_spots))
        storage_costs = np.random.randint(10, 60, size=(n_storage_units, n_spots))
        activation_costs = np.random.randint(30, 250, size=n_spots)

        # Capacities and demands
        transmission_capacity = np.random.randint(70, 250, size=n_spots)
        storage_capacity = np.random.randint(40, 200, size=n_spots)
        content_demand = np.random.randint(10, 25, size=n_contents)
        storage_demand = np.random.randint(5, 20, size=n_storage_units)

        # Distances for piecewise linear function
        distances = np.random.randint(1, 100, size=n_spots)

        res = {
            'n_contents': n_contents,
            'n_spots': n_spots,
            'n_storage_units': n_storage_units,
            'content_costs': content_costs,
            'storage_costs': storage_costs,
            'activation_costs': activation_costs,
            'transmission_capacity': transmission_capacity,
            'storage_capacity': storage_capacity,
            'content_demand': content_demand,
            'storage_demand': storage_demand,
            'distances': distances
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_contents = instance['n_contents']
        n_spots = instance['n_spots']
        n_storage_units = instance['n_storage_units']
        content_costs = instance['content_costs']
        storage_costs = instance['storage_costs']
        activation_costs = instance['activation_costs']
        transmission_capacity = instance['transmission_capacity']
        storage_capacity = instance['storage_capacity']
        content_demand = instance['content_demand']
        storage_demand = instance['storage_demand']
        distances = instance['distances']

        model = Model("MediaContentDistribution")

        # Variables
        m = {}
        h = {}
        for i in range(n_contents):
            for j in range(n_spots):
                m[i, j] = model.addVar(vtype="B", name=f"m_{i}_{j}")
                h[i, j] = model.addVar(vtype="B", name=f"h_{i}_{j}")

        s = {j: model.addVar(vtype="B", name=f"s_{j}") for j in range(n_spots)}

        n = {j: model.addVar(vtype="C", name=f"n_{j}") for j in range(n_spots)}

        # Objective function: Minimize total cost
        total_cost = quicksum(m[i, j] * content_costs[i, j] for i in range(n_contents) for j in range(n_spots)) + \
                     quicksum(h[i, j] * content_costs[i, j] * 0.1 for i in range(n_contents) for j in range(n_spots)) + \
                     quicksum(s[j] * activation_costs[j] for j in range(n_spots)) + \
                     quicksum(n[j] for j in range(n_spots))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_contents):
            model.addCons(quicksum(m[i, j] for j in range(n_spots)) == 1, name=f"content_assignment_{i}")
        
        # Capacities constraints
        for j in range(n_spots):
            model.addCons(quicksum(m[i, j] * content_demand[i] for i in range(n_contents)) <= transmission_capacity[j] * s[j],
                          name=f"transmission_capacity_{j}")

            model.addCons(quicksum(h[i, j] * content_demand[i] for i in range(n_contents)) <= storage_capacity[j] * s[j],
                          name=f"storage_capacity_{j}")

            # Piecewise linear penalty constraints
            model.addCons(n[j] >= distances[j] * s[j], name=f"piecewise_penalty_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_contents': 70,
        'max_contents': 600,
        'min_spots': 50,
        'max_spots': 300,
        'min_storage_units': 30,
        'max_storage_units': 300,
    }

    distribution = MediaContentDistribution(parameters, seed=seed)
    instance = distribution.generate_instance()
    solve_status, solve_time = distribution.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")