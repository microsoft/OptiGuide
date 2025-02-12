import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HubLocationProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        # Generate positions for nodes
        node_positions = np.random.rand(self.number_of_nodes, 2) * 100

        # Generate demand for each node
        demands = np.random.randint(self.min_demand, self.max_demand, self.number_of_nodes)

        # Calculate Euclidean distances between each pair of nodes
        distances = np.linalg.norm(node_positions[:, None] - node_positions, axis=2)

        res = {
            'node_positions': node_positions,
            'demands': demands,
            'distances': distances
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        node_positions = instance['node_positions']
        demands = instance['demands']
        distances = instance['distances']
        
        number_of_nodes = len(demands)
        
        model = Model("HubLocation")

        # Decision variables: y[j] = 1 if hub is placed at node j
        hub_selection = {
            j: model.addVar(vtype="B", name=f"y_{j}")
            for j in range(number_of_nodes)
        }

        # Decision variables: x[i, j] = 1 if node i is assigned to hub j
        node_assignment = {
            (i, j): model.addVar(vtype="B", name=f"x_{i}_{j}")
            for i in range(number_of_nodes)
            for j in range(number_of_nodes)
        }

        # Decision variables: c[j] = capacity of hub at node j
        hub_capacity = {
            j: model.addVar(vtype="I", name=f"c_{j}")
            for j in range(number_of_nodes)
        }

        # Objective: Minimize transportation cost
        objective_expr = quicksum(
            distances[i, j] * node_assignment[i, j] 
            for i in range(number_of_nodes) 
            for j in range(number_of_nodes)
        )

        # Constraints: Each node must be assigned to exactly one hub
        for i in range(number_of_nodes):
            model.addCons(
                quicksum(node_assignment[i, j] for j in range(number_of_nodes)) == 1,
                f"HubAssignment_{i}"
            )

        # Constraints: A node can only be assigned to an open hub
        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                model.addCons(
                    node_assignment[i, j] <= hub_selection[j],
                    f"NodeToHubAssignment_{i}_{j}"
                )

        # Constraints: Total demand assigned to a hub does not exceed its capacity
        for j in range(number_of_nodes):
            model.addCons(
                quicksum(demands[i] * node_assignment[i, j] for i in range(number_of_nodes)) <= hub_capacity[j],
                f"HubCapacityConstraint_{j}"
            )

        # Constraints: Stratify hub capacities based on open hubs
        for j in range(number_of_nodes):
            model.addCons(
                hub_capacity[j] <= self.max_hub_capacity * hub_selection[j],
                f"StratifiedHubCapacity_{j}"
            )

        # Constraints: Limit the number of hubs
        model.addCons(
            quicksum(hub_selection[j] for j in range(number_of_nodes)) <= self.max_number_of_hubs,
            f"MaxNumberOfHubsConstraint"
        )

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_nodes': 50,
        'max_number_of_hubs': 10,
        'min_demand': 25,
        'max_demand': 160,
        'max_hub_capacity': 600,
    }

    hub_problem = HubLocationProblem(parameters, seed=seed)
    instance = hub_problem.generate_instance()
    solve_status, solve_time = hub_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")