import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DataCenterPlacementOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_data_centers = random.randint(self.min_data_centers, self.max_data_centers)
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        # Cost matrices
        node_connection_costs = np.random.randint(50, 300, size=(num_nodes, num_data_centers))
        operational_costs = np.random.randint(1000, 5000, size=num_data_centers)

        # Node demands
        nodal_demand = np.random.randint(100, 500, size=num_nodes)

        # MegaServer capacity
        mega_server_capacity = np.random.randint(1000, 5000, size=num_data_centers)

        res = {
            'num_data_centers': num_data_centers,
            'num_nodes': num_nodes,
            'node_connection_costs': node_connection_costs,
            'operational_costs': operational_costs,
            'nodal_demand': nodal_demand,
            'mega_server_capacity': mega_server_capacity,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_data_centers = instance['num_data_centers']
        num_nodes = instance['num_nodes']
        node_connection_costs = instance['node_connection_costs']
        operational_costs = instance['operational_costs']
        nodal_demand = instance['nodal_demand']
        mega_server_capacity = instance['mega_server_capacity']

        model = Model("DataCenterPlacementOptimization")

        # Variables
        mega_server = {j: model.addVar(vtype="B", name=f"mega_server_{j}") for j in range(num_data_centers)}
        node_connection = {(i, j): model.addVar(vtype="B", name=f"node_connection_{i}_{j}") for i in range(num_nodes) for j in range(num_data_centers)}

        # Objective function: Minimize total costs
        total_cost = quicksum(node_connection[i, j] * node_connection_costs[i, j] for i in range(num_nodes) for j in range(num_data_centers)) + \
                     quicksum(mega_server[j] * operational_costs[j] for j in range(num_data_centers))
        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_nodes):
            model.addCons(quicksum(node_connection[i, j] for j in range(num_data_centers)) == 1, name=f"node_connection_{i}")

        # Logical constraints: A data center can only connect nodes if it has a mega server
        for j in range(num_data_centers):
            for i in range(num_nodes):
                model.addCons(node_connection[i, j] <= mega_server[j], name=f"data_center_node_{i}_{j}")

        # Clique Inequality Constraints for MegaServer Capacity
        for j in range(num_data_centers):
            cliques = self.find_cliques(num_nodes, nodal_demand, mega_server_capacity[j])
            for clique in cliques:
                model.addCons(quicksum(node_connection[i, j] * nodal_demand[i] for i in clique) <= mega_server_capacity[j], name=f"clique_mega_server_capacity_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

    ################# Helper Functions #################
    # Find clique sets for the current problem instance
    def find_cliques(self, num_nodes, nodal_demand, capacity):
        cliques = []
        temp_clique = []
        current_capacity = 0

        for i in range(num_nodes):
            if current_capacity + nodal_demand[i] <= capacity:
                temp_clique.append(i)
                current_capacity += nodal_demand[i]
            else:
                if temp_clique:
                    cliques.append(temp_clique)
                temp_clique = [i]
                current_capacity = nodal_demand[i]

        if temp_clique:
            cliques.append(temp_clique)

        return cliques

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_data_centers': 45,
        'max_data_centers': 70,
        'min_nodes': 100,
        'max_nodes': 600,
    }
    optimization = DataCenterPlacementOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")