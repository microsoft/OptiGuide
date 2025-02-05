import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class NetworkCommunicationScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_channels = random.randint(self.min_channels, self.max_channels)
        n_nodes = random.randint(self.min_nodes, self.max_nodes)
        n_tasks = random.randint(self.min_tasks, self.max_tasks)

        # Cost matrices
        channel_costs = np.random.randint(20, 150, size=(n_channels, n_nodes))
        uplink_costs = np.random.randint(30, 200, size=n_nodes)
        activation_levels = np.random.randint(15, 75, size=n_nodes)

        # Capacities and demands
        node_capacity = np.random.randint(50, 300, size=n_nodes)
        task_demand = np.random.randint(5, 25, size=n_channels)

        res = {
            'n_channels': n_channels,
            'n_nodes': n_nodes,
            'n_tasks': n_tasks,
            'channel_costs': channel_costs,
            'uplink_costs': uplink_costs,
            'activation_levels': activation_levels,
            'node_capacity': node_capacity,
            'task_demand': task_demand
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_channels = instance['n_channels']
        n_nodes = instance['n_nodes']
        n_tasks = instance['n_tasks']
        channel_costs = instance['channel_costs']
        uplink_costs = instance['uplink_costs']
        activation_levels = instance['activation_levels']
        node_capacity = instance['node_capacity']
        task_demand = instance['task_demand']

        model = Model("NetworkCommunicationScheduling")

        # Variables
        c = {}
        for i in range(n_channels):
            for j in range(n_nodes):
                c[i, j] = model.addVar(vtype="B", name=f"c_{i}_{j}")
        
        u = {j: model.addVar(vtype="B", name=f"u_{j}") for j in range(n_nodes)}
        
        a = {j: model.addVar(vtype="C", name=f"a_{j}") for j in range(n_nodes)}

        # Objective function: Minimize total communication cost
        total_cost = quicksum(c[i, j] * channel_costs[i, j] for i in range(n_channels) for j in range(n_nodes)) + \
                     quicksum(u[j] * uplink_costs[j] for j in range(n_nodes)) + \
                     quicksum(a[j] * activation_levels[j] for j in range(n_nodes))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_channels):
            model.addCons(quicksum(c[i, j] for j in range(n_nodes)) == 1, name=f"channel_assignment_{i}")
        
        # Network capacity constraints
        for j in range(n_nodes):
            model.addCons(quicksum(c[i, j] * task_demand[i] for i in range(n_channels)) <= node_capacity[j] * u[j],
                          name=f"node_capacity_{j}")

            model.addCons(quicksum(a[j] for j in range(n_nodes)) >= u[j], name=f"activation_level_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_channels': 2,
        'max_channels': 100,
        'min_nodes': 7,
        'max_nodes': 2000,
        'min_tasks': 1,
        'max_tasks': 10,
    }

    network_scheduler = NetworkCommunicationScheduling(parameters, seed=seed)
    instance = network_scheduler.generate_instance()
    solve_status, solve_time = network_scheduler.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")