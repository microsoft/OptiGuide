import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class NetworkSafetyOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_sensors = random.randint(self.min_sensors, self.max_sensors)
        n_nodes = random.randint(self.min_nodes, self.max_nodes)
        n_tasks = random.randint(self.min_tasks, self.max_tasks)

        # Cost matrices
        sensor_costs = np.random.randint(20, 150, size=(n_sensors, n_nodes))
        penalty_costs = np.random.randint(30, 200, size=n_nodes)
        safety_levels = np.random.randint(10, 50, size=n_sensors)

        # Capacities and requirements
        node_capacity = np.random.randint(50, 300, size=n_nodes)
        sensor_demand = np.random.randint(5, 25, size=n_sensors)

        res = {
            'n_sensors': n_sensors,
            'n_nodes': n_nodes,
            'n_tasks': n_tasks,
            'sensor_costs': sensor_costs,
            'penalty_costs': penalty_costs,
            'safety_levels': safety_levels,
            'node_capacity': node_capacity,
            'sensor_demand': sensor_demand
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_sensors = instance['n_sensors']
        n_nodes = instance['n_nodes']
        n_tasks = instance['n_tasks']
        sensor_costs = instance['sensor_costs']
        penalty_costs = instance['penalty_costs']
        safety_levels = instance['safety_levels']
        node_capacity = instance['node_capacity']
        sensor_demand = instance['sensor_demand']

        model = Model("NetworkSafetyOptimization")

        # Variables
        nsv = {}
        for i in range(n_sensors):
            for j in range(n_nodes):
                nsv[i, j] = model.addVar(vtype="B", name=f"nsv_{i}_{j}")
        
        mpv = {j: model.addVar(vtype="C", name=f"mpv_{j}") for j in range(n_nodes)}
        
        ccv = {j: model.addVar(vtype="C", name=f"ccv_{j}") for j in range(n_nodes)}

        # Objective function: Minimize total safety and penalty cost
        total_cost = quicksum(nsv[i, j] * sensor_costs[i, j] for i in range(n_sensors) for j in range(n_nodes)) + \
                     quicksum(mpv[j] * penalty_costs[j] for j in range(n_nodes)) + \
                     quicksum(ccv[j] * safety_levels[j%n_sensors] for j in range(n_nodes))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_sensors):
            model.addCons(quicksum(nsv[i, j] for j in range(n_nodes)) == 1, name=f"sensor_assignment_{i}")
        
        # Node capacity constraints with safety penalty
        for j in range(n_nodes):
            model.addCons(quicksum(nsv[i, j] * sensor_demand[i] for i in range(n_sensors)) <= node_capacity[j],
                          name=f"node_capacity_{j}")

            model.addCons(mpv[j] <= node_capacity[j], name=f"node_penalty_{j}")

            model.addCons(quicksum(ccv[j] for j in range(n_nodes)) >= quicksum(safety_levels[i] * nsv[i, j] for i in range(n_sensors)),
                          name=f"minimum_safety_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 123
    parameters = {
        'min_sensors': 40,
        'max_sensors': 200,
        'min_nodes': 70,
        'max_nodes': 1000,
        'min_tasks': 10,
        'max_tasks': 50,
    }

    network_safety_optimizer = NetworkSafetyOptimization(parameters, seed=seed)
    instance = network_safety_optimizer.generate_instance()
    solve_status, solve_time = network_safety_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")