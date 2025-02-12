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
        n_timeslots = random.randint(self.min_timeslots, self.max_timeslots)
        
        # Cost matrices
        sensor_costs = np.random.randint(20, 150, size=(n_sensors, n_nodes))
        penalty_costs = np.random.randint(30, 200, size=n_nodes)
        safety_levels = np.random.randint(10, 50, size=n_sensors)
        timeslot_costs = np.random.randint(10, 100, size=(n_sensors, n_timeslots))

        # Capacities and requirements
        node_capacity = np.random.randint(50, 300, size=n_nodes)
        sensor_demand = np.random.randint(5, 25, size=n_sensors)

        res = {
            'n_sensors': n_sensors,
            'n_nodes': n_nodes,
            'n_tasks': n_tasks,
            'n_timeslots': n_timeslots,
            'sensor_costs': sensor_costs,
            'penalty_costs': penalty_costs,
            'safety_levels': safety_levels,
            'node_capacity': node_capacity,
            'sensor_demand': sensor_demand,
            'timeslot_costs': timeslot_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_sensors = instance['n_sensors']
        n_nodes = instance['n_nodes']
        n_tasks = instance['n_tasks']
        n_timeslots = instance['n_timeslots']
        sensor_costs = instance['sensor_costs']
        penalty_costs = instance['penalty_costs']
        safety_levels = instance['safety_levels']
        node_capacity = instance['node_capacity']
        sensor_demand = instance['sensor_demand']
        timeslot_costs = instance['timeslot_costs']

        model = Model("NetworkSafetyOptimization")

        # Variables
        nsv = {}
        for i in range(n_sensors):
            for j in range(n_nodes):
                nsv[i, j] = model.addVar(vtype="B", name=f"nsv_{i}_{j}")

        mpv = {j: model.addVar(vtype="C", name=f"mpv_{j}") for j in range(n_nodes)}
        
        ccv = {j: model.addVar(vtype="C", name=f"ccv_{j}") for j in range(n_nodes)}

        # Time slot usage
        tsv = {}
        for i in range(n_sensors):
            for t in range(n_timeslots):
                tsv[i, t] = model.addVar(vtype="B", name=f"tsv_{i}_{t}")

        # Objective function: Minimize total safety and penalty cost
        total_cost = quicksum(nsv[i, j] * sensor_costs[i, j] for i in range(n_sensors) for j in range(n_nodes)) + \
                     quicksum(mpv[j] * penalty_costs[j] for j in range(n_nodes)) + \
                     quicksum(ccv[j] * safety_levels[j % n_sensors] for j in range(n_nodes)) + \
                     quicksum(tsv[i, t] * timeslot_costs[i, t] for i in range(n_sensors) for t in range(n_timeslots))

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
            
        # Time slot constraints using Big M formulation
        M = 1e6
        for i in range(n_sensors):
            for t in range(n_timeslots):
                model.addCons(quicksum(nsv[i, j] for j in range(n_nodes)) - M * tsv[i, t] <= 0, name=f"active_time_{i}_{t}")
                model.addCons(tsv[i, t] <= quicksum(nsv[i, j] for j in range(n_nodes)), name=f"time_slot_usage_{i}_{t}")

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
        'min_timeslots': 10,
        'max_timeslots': 24,
    }

    network_safety_optimizer = NetworkSafetyOptimization(parameters, seed=seed)
    instance = network_safety_optimizer.generate_instance()
    solve_status, solve_time = network_safety_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")