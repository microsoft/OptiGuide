import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class NetworkDesignOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_machines = random.randint(self.min_machines, self.max_machines)
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        # Cost matrices
        connection_costs = np.random.randint(10, 100, size=(num_nodes, num_machines))
        activation_costs = np.random.randint(500, 2000, size=num_machines)
        congestion_costs = np.random.randint(50, 300, size=num_machines)

        # Network machine capacities
        machine_capacities = np.random.randint(10, 50, size=num_machines)

        # Priority nodes
        priorities = np.random.randint(0, 2, size=num_nodes)

        res = {
            'num_machines': num_machines,
            'num_nodes': num_nodes,
            'connection_costs': connection_costs,
            'activation_costs': activation_costs,
            'congestion_costs': congestion_costs,
            'machine_capacities': machine_capacities,
            'priorities': priorities,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_machines = instance['num_machines']
        num_nodes = instance['num_nodes']
        connection_costs = instance['connection_costs']
        activation_costs = instance['activation_costs']
        congestion_costs = instance['congestion_costs']
        machine_capacities = instance['machine_capacities']
        priorities = instance['priorities']

        model = Model("NetworkDesignOptimization")

        # Variables
        machine_active = {j: model.addVar(vtype="B", name=f"machine_active_{j}") for j in range(num_machines)}
        connection = {(i, j): model.addVar(vtype="B", name=f"connection_{i}_{j}") for i in range(num_nodes) for j in range(num_machines)}
        high_priority_node = {i: model.addVar(vtype="B", name=f"high_priority_node_{i}") for i in range(num_nodes)}

        # Objective function: Minimize total costs including congestion costs
        total_cost = quicksum(connection[i, j] * connection_costs[i, j] for i in range(num_nodes) for j in range(num_machines)) + \
                     quicksum(machine_active[j] * activation_costs[j] for j in range(num_machines)) + \
                     quicksum(machine_active[j] * congestion_costs[j] for j in range(num_machines))

        model.setObjective(total_cost, "minimize")

        # Constraints
        # Each node must be connected to exactly one machine
        for i in range(num_nodes):
            model.addCons(quicksum(connection[i, j] for j in range(num_machines)) == 1, name=f"node_connection_{i}")

        # Logical constraints: A node can only be connected to a machine if the machine is active
        for j in range(num_machines):
            for i in range(num_nodes):
                model.addCons(connection[i, j] <= machine_active[j], name=f"machine_connection_constraint_{i}_{j}")

        # Capacity constraints: A machine can only handle up to its capacity
        for j in range(num_machines):
            model.addCons(quicksum(connection[i, j] for i in range(num_nodes)) <= machine_capacities[j], name=f"machine_capacity_{j}")
        
        # New constraint: Limit on total number of active machines
        model.addCons(quicksum(machine_active[j] for j in range(num_machines)) <= self.max_active_machines, name="max_active_machines")

        # New constraints: High-priority nodes can only be connected to high-performance machines
        for i in range(num_nodes):
            if priorities[i] == 1:
                model.addCons(high_priority_node[i] == 1, name=f"priority_node_{i}")

        min_high_priority_nodes = int(self.min_priority_ratio * num_nodes)
        model.addCons(quicksum(high_priority_node[i] for i in range(num_nodes)) >= min_high_priority_nodes, name="min_high_priority_nodes")

        model.optimize()

        solution = {"connections": {}, "machines": {}}
        if model.getStatus() == "optimal":
            for j in range(num_machines):
                solution["machines"][j] = model.getVal(machine_active[j])
            for i in range(num_nodes):
                for j in range(num_machines):
                    if model.getVal(connection[i, j]) > 0.5:
                        solution["connections"][i] = j

        return solution, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_machines': 200,
        'max_machines': 480,
        'min_nodes': 11,
        'max_nodes': 1600,
        'max_active_machines': 600,
        'min_priority_ratio': 0.45,
    }

    optimization = NetworkDesignOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solution, total_cost = optimization.solve(instance)

    print(f"Solution: {solution}")
    print(f"Total Cost: {total_cost:.2f}")