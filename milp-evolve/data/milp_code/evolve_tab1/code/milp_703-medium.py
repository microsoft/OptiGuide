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
        operational_costs_type1 = np.random.randint(1000, 3000, size=num_data_centers)
        operational_costs_type2 = np.random.randint(2500, 5000, size=num_data_centers)

        # Node demands
        nodal_demand = np.random.randint(100, 500, size=num_nodes)

        # Server capacities for different types
        server_capacity_type1 = np.random.randint(1000, 3000, size=num_data_centers)
        server_capacity_type2 = np.random.randint(3000, 5000, size=num_data_centers)

        # Distances between nodes and data centers
        distances = np.random.randint(1, 100, size=(num_nodes, num_data_centers))

        # Piecewise linear energy costs
        energy_cost_breakpoints = np.sort(np.random.randint(1, 100, size=4))
        energy_cost_values = np.sort(np.random.randint(100, 1000, size=5))

        res = {
            'num_data_centers': num_data_centers,
            'num_nodes': num_nodes,
            'node_connection_costs': node_connection_costs,
            'operational_costs_type1': operational_costs_type1,
            'operational_costs_type2': operational_costs_type2,
            'nodal_demand': nodal_demand,
            'server_capacity_type1': server_capacity_type1,
            'server_capacity_type2': server_capacity_type2,
            'distances': distances,
            'energy_cost_breakpoints': energy_cost_breakpoints,
            'energy_cost_values': energy_cost_values,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_data_centers = instance['num_data_centers']
        num_nodes = instance['num_nodes']
        node_connection_costs = instance['node_connection_costs']
        operational_costs_type1 = instance['operational_costs_type1']
        operational_costs_type2 = instance['operational_costs_type2']
        nodal_demand = instance['nodal_demand']
        server_capacity_type1 = instance['server_capacity_type1']
        server_capacity_type2 = instance['server_capacity_type2']
        distances = instance['distances']
        energy_cost_breakpoints = instance['energy_cost_breakpoints']
        energy_cost_values = instance['energy_cost_values']

        model = Model("DataCenterPlacementOptimization")

        # Variables
        mega_server_type1 = {j: model.addVar(vtype="B", name=f"mega_server_type1_{j}") for j in range(num_data_centers)}
        mega_server_type2 = {j: model.addVar(vtype="B", name=f"mega_server_type2_{j}") for j in range(num_data_centers)}
        node_connection = {(i, j): model.addVar(vtype="B", name=f"node_connection_{i}_{j}") for i in range(num_nodes) for j in range(num_data_centers)}
        server_allocations = {(i, j): model.addVar(vtype="I", name=f"server_allocations_{i}_{j}") for i in range(num_nodes) for j in range(num_data_centers)}
        
        # Additional variables for piecewise linear energy cost
        energy_cost_vars = {j: model.addVar(vtype="C", name=f"energy_cost_{j}") for j in range(num_data_centers)}

        # Objective function: Minimize total costs
        total_cost = quicksum(node_connection[i, j] * node_connection_costs[i, j] for i in range(num_nodes) for j in range(num_data_centers)) + \
                     quicksum(mega_server_type1[j] * operational_costs_type1[j] for j in range(num_data_centers)) + \
                     quicksum(mega_server_type2[j] * operational_costs_type2[j] for j in range(num_data_centers)) + \
                     quicksum(node_connection[i, j] * distances[i, j] for i in range(num_nodes) for j in range(num_data_centers)) + \
                     quicksum(energy_cost_vars[j] for j in range(num_data_centers))  # Adding piecewise linear energy costs

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_nodes):
            model.addCons(quicksum(node_connection[i, j] for j in range(num_data_centers)) == 1, name=f"node_connection_{i}")

        for j in range(num_data_centers):
            for i in range(num_nodes):
                model.addCons(node_connection[i, j] <= mega_server_type1[j] + mega_server_type2[j], name=f"data_center_node_{i}_{j}")
                
        for j in range(num_data_centers):
            model.addCons(quicksum(nodal_demand[i] * node_connection[i, j] for i in range(num_nodes)) <= 
                          mega_server_type1[j] * server_capacity_type1[j] + 
                          mega_server_type2[j] * server_capacity_type2[j], name=f"capacity_{j}")

        # New Constraints: Piecewise linear energy cost constraints
        for j in range(num_data_centers):
            server_load = quicksum(server_allocations[i, j] for i in range(num_nodes))
            for k in range(len(energy_cost_breakpoints) - 1):
                model.addCons(energy_cost_vars[j] >= energy_cost_values[k] * (server_load - energy_cost_breakpoints[k]) / (energy_cost_breakpoints[k + 1] - energy_cost_breakpoints[k]),
                              name=f"piecewise_lower_{j}_{k}")
                model.addCons(energy_cost_vars[j] <= energy_cost_values[k + 1] * (server_load - energy_cost_breakpoints[k]) / (energy_cost_breakpoints[k + 1] - energy_cost_breakpoints[k]),
                              name=f"piecewise_upper_{j}_{k}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_data_centers': 180,
        'max_data_centers': 1400,
        'min_nodes': 12,
        'max_nodes': 360,
    }

    optimization = DataCenterPlacementOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")