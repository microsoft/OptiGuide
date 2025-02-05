import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NetworkOptimizationProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################

    def generate_instance(self):
        # Generating installation costs for routers
        router_install_cost = np.random.randint(self.min_router_install_cost, self.max_router_install_cost, self.number_of_nodes)
        
        # Generating operational connection costs between nodes
        connection_costs = np.random.randint(self.min_connection_cost, self.max_connection_cost, (self.number_of_nodes, self.number_of_nodes))
        
        # Generating router capacities
        router_capacities = np.random.randint(self.min_router_capacity, self.max_router_capacity, self.number_of_nodes)

        # Generating link capacities between routers
        link_capacities = np.random.randint(self.min_link_capacity, self.max_link_capacity, (self.number_of_nodes, self.number_of_nodes))

        # Signal range matrix
        distances = np.random.rand(self.number_of_nodes, self.number_of_nodes)
        signal_ranges = np.where(distances <= self.max_signal_range, 1, 0)
        
        res = {
            'router_install_cost': router_install_cost,
            'connection_costs': connection_costs,
            'router_capacities': router_capacities,
            'link_capacities': link_capacities,
            'signal_ranges': signal_ranges,
        }

        return res

    ################# PySCIPOpt Modeling #################

    def solve(self, instance):
        router_install_cost = instance['router_install_cost']
        connection_costs = instance['connection_costs']
        router_capacities = instance['router_capacities']
        link_capacities = instance['link_capacities']
        signal_ranges = instance['signal_ranges']

        number_of_nodes = len(router_install_cost)

        model = Model("NetworkOptimizationProblem")

        # Decision variables
        router_installation = {i: model.addVar(vtype="B", name=f"router_installation_{i}") for i in range(number_of_nodes)}
        link_installation = {(i, j): model.addVar(vtype="B", name=f"link_installation_{i}_{j}") for i in range(number_of_nodes) for j in range(number_of_nodes)}

        # Objective: Minimize total cost (installation costs + connection costs)
        objective_expr = quicksum(router_install_cost[i] * router_installation[i] for i in range(number_of_nodes))
        objective_expr += quicksum(connection_costs[i][j] * link_installation[(i, j)] for i in range(number_of_nodes) for j in range(number_of_nodes))

        model.setObjective(objective_expr, "minimize")

        # Constraint: Each node must be covered by at least one router within signal range
        for i in range(number_of_nodes):
            model.addCons(quicksum(router_installation[j] * signal_ranges[i][j] for j in range(number_of_nodes)) >= 1, f"NodeCoverage_{i}")

        # Constraint: Router capacity constraints
        for i in range(number_of_nodes):
            model.addCons(quicksum(link_installation[(i, j)] for j in range(number_of_nodes)) <= router_capacities[i], f"RouterCapacity_{i}")

        # Constraint: Link capacity constraints
        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                model.addCons(link_installation[(i, j)] <= link_capacities[i][j], f"LinkCapacity_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_nodes': 300,
        'min_router_install_cost': 1000,
        'max_router_install_cost': 3000,
        'min_connection_cost': 20,
        'max_connection_cost': 800,
        'min_router_capacity': 5,
        'max_router_capacity': 400,
        'min_link_capacity': 450,
        'max_link_capacity': 600,
        'max_signal_range': 0.55,
    }
    
    network_optimization = NetworkOptimizationProblem(parameters, seed=seed)
    instance = network_optimization.generate_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")