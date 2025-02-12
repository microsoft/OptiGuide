import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HealthcareResourceOptimization:
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

        # Transportation costs and traffic conditions
        transport_costs = np.random.randint(20, 100, size=(num_nodes, num_nodes))
        traffic_conditions = np.random.randint(1, 10, size=(num_nodes, num_nodes))

        # Emergency response
        emergency_response_times = np.random.randint(5, 30, size=num_nodes)

        res = {
            'num_data_centers': num_data_centers,
            'num_nodes': num_nodes,
            'node_connection_costs': node_connection_costs,
            'operational_costs': operational_costs,
            'nodal_demand': nodal_demand,
            'mega_server_capacity': mega_server_capacity,
            'transport_costs': transport_costs,
            'traffic_conditions': traffic_conditions,
            'emergency_response_times': emergency_response_times,
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
        transport_costs = instance['transport_costs']
        traffic_conditions = instance['traffic_conditions']
        emergency_response_times = instance['emergency_response_times']

        model = Model("HealthcareResourceOptimization")

        ##########################################
        ## Variables ##
        ##########################################

        # Mega server binary variable
        mega_server = {j: model.addVar(vtype="B", name=f"mega_server_{j}") for j in range(num_data_centers)}

        # Node connection binary variable
        node_connection = {(i, j): model.addVar(vtype="B", name=f"node_connection_{i}_{j}") for i in range(num_nodes) for j in range(num_data_centers)}

        # Transportation binary variable
        transportation = {(i, k): model.addVar(vtype="B", name=f"transportation_{i}_{k}") for i in range(num_nodes) for k in range(num_nodes)}

        # Transportation time variable
        transport_time = {(i, k): model.addVar(vtype="C", name=f"transport_time_{i}_{k}") for i in range(num_nodes) for k in range(num_nodes)}

        # Emergency team binary variable
        emergency_team = {i: model.addVar(vtype="B", name=f"emergency_team_{i}") for i in range(num_nodes)}

        ##########################################
        ## Objective Function: Minimize Total Costs and Response Time ##
        ##########################################

        total_cost = (quicksum(node_connection[i, j] * node_connection_costs[i, j]
                               for i in range(num_nodes) for j in range(num_data_centers))
                      + quicksum(mega_server[j] * operational_costs[j] for j in range(num_data_centers))
                      + quicksum(transportation[i, k] * transport_costs[i, k]
                                 for i in range(num_nodes) for k in range(num_nodes))
                      + quicksum(emergency_team[i] * emergency_response_times[i] for i in range(num_nodes)))

        model.setObjective(total_cost, "minimize")

        ##########################################
        ## Constraints ##
        ##########################################

        # Each node should be connected to exactly one data center
        for i in range(num_nodes):
            model.addCons(quicksum(node_connection[i, j] for j in range(num_data_centers)) == 1, name=f"node_connection_{i}")

        # A data center can only connect nodes if it has a mega server
        for j in range(num_data_centers):
            for i in range(num_nodes):
                model.addCons(node_connection[i, j] <= mega_server[j], name=f"data_center_node_{i}_{j}")

        # MegaServer capacity constraints
        for j in range(num_data_centers):
            model.addCons(quicksum(node_connection[i, j] * nodal_demand[i] for i in range(num_nodes)) <= mega_server_capacity[j], name=f"mega_server_capacity_{j}")

        # Transportation capacity constraints
        for i in range(num_nodes):
            for k in range(num_nodes):
                model.addCons(transport_time[i, k] == transportation[i, k] * traffic_conditions[i, k], name=f"transport_time_constraint_{i}_{k}")

        # Ensure minimal downtime
        for i in range(num_nodes):
            model.addCons(quicksum(transport_time[i, k] for k in range(num_nodes)) <= self.max_transport_time, name=f"downtime_minimization_{i}")

        # Emergency team allocation
        model.addCons(quicksum(emergency_team[i] for i in range(num_nodes)) >= self.min_emergency_teams, name="min_emergency_teams")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_data_centers': 75,
        'max_data_centers': 600,
        'min_nodes': 37,
        'max_nodes': 900,
        'max_transport_time': 250,
        'min_emergency_teams': 25,
    }

    optimization = HealthcareResourceOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")