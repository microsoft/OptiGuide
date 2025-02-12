import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class TelecommunicationNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_network_costs(self, n_nodes, max_cost):
        return np.random.rand(n_nodes, n_nodes) * max_cost

    def generate_instance(self):
        traffic_demand = self.randint(self.n_nodes, self.traffic_demand_interval)
        hub_capacities = self.randint(self.n_hubs, self.capacity_interval)
        installation_cost = self.randint(self.n_hubs, self.installation_cost_interval)
        network_costs = self.generate_network_costs(self.n_nodes, self.max_network_cost)

        res = {
            'traffic_demand': traffic_demand,
            'hub_capacities': hub_capacities,
            'installation_cost': installation_cost,
            'network_costs': network_costs
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        traffic_demand = instance['traffic_demand']
        hub_capacities = instance['hub_capacities']
        installation_cost = instance['installation_cost']
        network_costs = instance['network_costs']

        n_nodes = len(traffic_demand)
        n_hubs = len(hub_capacities)

        model = Model("TelecommunicationNetworkOptimization")
        
        # Decision variables
        open_hubs = {j: model.addVar(vtype="B", name=f"OpenHub_{j}") for j in range(n_hubs)}
        allocate_traffic = {(i, j): model.addVar(vtype="C", name=f"Allocate_{i}_{j}") for i in range(n_nodes) for j in range(n_hubs)}
        connect_hubs = {(j, k): model.addVar(vtype="B", name=f"Connect_{j}_{k}") for j in range(n_hubs) for k in range(n_hubs) if j != k}
        
        # Objective: Minimize total cost (installation + network establishment)
        total_installation_cost = quicksum(installation_cost[j] * open_hubs[j] for j in range(n_hubs))
        total_network_cost = quicksum(network_costs[j, k] * connect_hubs[j, k] for j in range(n_hubs) for k in range(n_hubs) if j != k)
        
        model.setObjective(total_installation_cost + total_network_cost, "minimize")

        # Constraints: Traffic demand satisfaction
        for i in range(n_nodes):
            model.addCons(quicksum(allocate_traffic[i, j] for j in range(n_hubs)) >= traffic_demand[i], f"TrafficDemand_{i}")

        # Constraints: Hub capacity limits
        for j in range(n_hubs):
            model.addCons(quicksum(allocate_traffic[i, j] for i in range(n_nodes)) <= hub_capacities[j] * open_hubs[j], f"HubCapacity_{j}")
        
        # Connectivity constraints: Ensure hubs are only connected if both are open
        for j in range(n_hubs):
            for k in range(n_hubs):
                if j != k:
                    model.addCons(connect_hubs[j, k] <= open_hubs[j], f"Connectivity1_{j}_{k}")
                    model.addCons(connect_hubs[j, k] <= open_hubs[k], f"Connectivity2_{j}_{k}")

        # Constraint: a limit on the number of hubs that can be opened
        model.addCons(quicksum(open_hubs[j] for j in range(n_hubs)) <= self.hub_limit, "HubLimit")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 500,
        'n_hubs': 50,
        'traffic_demand_interval': (100, 400),
        'capacity_interval': (1000, 5000),
        'installation_cost_interval': (10000, 50000),
        'max_network_cost': 100,
        'hub_limit': 175,
    }

    network_optimization = TelecommunicationNetworkOptimization(parameters, seed=seed)
    instance = network_optimization.generate_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")