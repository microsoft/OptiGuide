import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class LogisticsWarehouseOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_latency_matrix(self, n_hubs, max_latency):
        return np.random.rand(n_hubs, n_hubs) * max_latency

    def generate_instance(self):
        node_traffic = self.randint(self.num_nodes, self.traffic_demand_interval)
        hub_capacities = self.randint(self.num_hubs, self.capacity_interval)
        installation_cost = self.randint(self.num_hubs, self.installation_cost_interval)
        latency_matrix = self.generate_latency_matrix(self.num_hubs, self.latency_cost)

        res = {
            'node_traffic': node_traffic,
            'hub_capacities': hub_capacities,
            'installation_cost': installation_cost,
            'latency_matrix': latency_matrix
        }

        ### instance data generation end ###
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        node_traffic = instance['node_traffic']
        hub_capacities = instance['hub_capacities']
        installation_cost = instance['installation_cost']
        latency_matrix = instance['latency_matrix']

        num_nodes = len(node_traffic)
        num_hubs = len(hub_capacities)

        model = Model("LogisticsWarehouseOptimization")
        
        # Decision variables
        open_hubs = {j: model.addVar(vtype="B", name=f"OpenHub_{j}") for j in range(num_hubs)}
        allocate_traffic = {(i, j): model.addVar(vtype="C", name=f"TrafficAllocation_{i}_{j}") for i in range(num_nodes) for j in range(num_hubs)}
        connect_hubs = {(j, k): model.addVar(vtype="B", name=f"HubConnection_{j}_{k}") for j in range(num_hubs) for k in range(num_hubs) if j != k}
        
        # Objective: Minimize total cost (installation + latency)
        total_installation_cost = quicksum(installation_cost[j] * open_hubs[j] for j in range(num_hubs))
        total_latency_cost = quicksum(latency_matrix[j, k] * connect_hubs[j, k] for j in range(num_hubs) for k in range(num_hubs) if j != k)
        
        model.setObjective(total_installation_cost + total_latency_cost, "minimize")

        # Constraints: Traffic demand satisfaction
        for i in range(num_nodes):
            model.addCons(quicksum(allocate_traffic[i, j] for j in range(num_hubs)) >= node_traffic[i], f"NodeCapacity_{i}")

        # Constraints: Hub capacity limits
        for j in range(num_hubs):
            model.addCons(quicksum(allocate_traffic[i, j] for i in range(num_nodes)) <= hub_capacities[j] * open_hubs[j], f"MaxHubUtilization_{j}")
        
        # Connectivity constraints: Ensure hubs are only connected if both are open
        for j in range(num_hubs):
            for k in range(num_hubs):
                if j != k:
                    model.addCons(connect_hubs[j, k] <= open_hubs[j], f"ConnectivityConstraint1_{j}_{k}")
                    model.addCons(connect_hubs[j, k] <= open_hubs[k], f"ConnectivityConstraint2_{j}_{k}")

        # Constraints: Shipping latency limit
        for j in range(num_hubs):
            for k in range(num_hubs):
                if j != k:
                    model.addCons(latency_matrix[j, k] * connect_hubs[j, k] <= self.latency_limit, f"NetworkLatency_{j}_{k}")

        # Constraint: a limit on the number of hubs that can be opened
        model.addCons(quicksum(open_hubs[j] for j in range(num_hubs)) <= self.max_hubs, "MaxHubs")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_nodes': 500,
        'num_hubs': 50,
        'traffic_demand_interval': (100, 400),
        'capacity_interval': (1000, 5000),
        'installation_cost_interval': (10000, 50000),
        'latency_cost': 100,
        'latency_limit': 200,
        'max_hubs': 175,
    }

    logistics_optimization = LogisticsWarehouseOptimization(parameters, seed=seed)
    instance = logistics_optimization.generate_instance()
    solve_status, solve_time = logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")