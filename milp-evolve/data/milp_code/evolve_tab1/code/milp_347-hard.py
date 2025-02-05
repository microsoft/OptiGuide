import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class CloudServerAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data generation #################
    def get_instance(self):
        assert self.min_bandwidth >= 0 and self.max_bandwidth >= self.min_bandwidth
        assert self.server_min_capacity >= 0 and self.server_max_capacity >= self.server_min_capacity

        bandwidth_demands = self.min_bandwidth + (self.max_bandwidth - self.min_bandwidth) * np.random.rand(self.n_flows)
        link_costs = []

        while len(link_costs) < self.n_links:
            path_length = np.random.randint(1, self.max_path_length + 1)
            path = np.random.choice(self.n_servers, size=path_length, replace=False)
            cost = max(bandwidth_demands[path].sum() + np.random.normal(0, 10), 0)
            traffic = np.random.poisson(lam=5)

            link_costs.append((path.tolist(), cost, traffic))

        traffic_per_server = [[] for _ in range(self.n_servers)]
        for i, link in enumerate(link_costs):
            path, cost, traffic = link
            for server in path:
                traffic_per_server[server].append(i)

        server_capacity = np.random.uniform(self.server_min_capacity, self.server_max_capacity, size=self.n_servers).tolist()
        host_cost = np.random.uniform(10, 50, size=self.n_servers).tolist()
        traffic_cost = np.random.uniform(5, 20, size=len(link_costs)).tolist()
        maintenance_budget = np.random.randint(5000, 10000)
        
        congestion_penalty_coeff = np.random.uniform(100, 500)
        minimum_bandwidth = np.random.uniform(100, 120)

        return {
            "link_costs": link_costs,
            "traffic_per_server": traffic_per_server,
            "server_capacity": server_capacity,
            "host_cost": host_cost,
            "traffic_cost": traffic_cost,
            "maintenance_budget": maintenance_budget,
            "congestion_penalty_coeff": congestion_penalty_coeff,
            "minimum_bandwidth": minimum_bandwidth
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        link_costs = instance['link_costs']
        traffic_per_server = instance['traffic_per_server']
        server_capacity = instance['server_capacity']
        host_cost = instance['host_cost']
        traffic_cost = instance['traffic_cost']
        maintenance_budget = instance['maintenance_budget']
        congestion_penalty_coeff = instance['congestion_penalty_coeff']
        minimum_bandwidth = instance['minimum_bandwidth']

        model = Model("CloudServerAllocation")

        link_vars = {i: model.addVar(vtype="B", name=f"Link_{i}") for i in range(len(link_costs))}
        server_vars = {j: model.addVar(vtype="B", name=f"Server_{j}") for j in range(len(server_capacity))}
        b_vars = {(i, j): model.addVar(vtype="C", name=f"b_{i}_{j}", lb=0, ub=1) for i in range(len(link_costs)) for j in range(len(server_capacity))}
        server_load = {j: model.addVar(vtype="I", name=f"load_{j}", lb=0) for j in range(len(server_capacity))}
        congestion_vars = {j: model.addVar(vtype="I", name=f"congestion_{j}", lb=0) for j in range(len(server_capacity))}
        minimum_bandwidth_vars = {i: model.addVar(vtype="B", name=f"MinBandwidth_{i}") for i in range(len(link_costs))}

        objective_expr = quicksum(cost * link_vars[i] for i, (path, cost, traffic) in enumerate(link_costs)) \
                         - quicksum(host_cost[j] * server_vars[j] for j in range(len(server_capacity))) \
                         - quicksum(traffic_cost[i] * b_vars[i, j] for i in range(len(link_costs)) for j in range(len(server_capacity))) \
                         - quicksum(congestion_penalty_coeff * congestion_vars[j] for j in range(len(server_capacity))) \
                         + quicksum(minimum_bandwidth * minimum_bandwidth_vars[i] for i in range(len(link_costs)))

        # Constraints: Each traffic flow can only be part of one accepted link
        for server, link_indices in enumerate(traffic_per_server):
            model.addCons(quicksum(link_vars[link_idx] for link_idx in link_indices) <= 1, f"Traffic_{server}")

        # Link assignment to server
        for i in range(len(link_costs)):
            model.addCons(quicksum(b_vars[i, j] for j in range(len(server_capacity))) == link_vars[i], f"LinkServer_{i}")

        # Server capacity constraints
        for j in range(len(server_capacity)):
            model.addCons(quicksum(b_vars[i, j] for i in range(len(link_costs))) <= server_capacity[j] * server_vars[j], f"ServerCapacity_{j}")

        # Server load constraints
        for j in range(len(server_capacity)):
            model.addCons(server_load[j] == quicksum(b_vars[i, j] * link_costs[i][2] for i in range(len(link_costs))), f"Load_{j}")

        # Penalty for server congestion
        for j in range(len(server_capacity)):
            model.addCons(congestion_vars[j] >= server_load[j] - server_capacity[j], f"Congestion_{j}")

        # Minimum bandwidth constraints
        for i in range(len(link_costs)):
            if link_costs[i][1] > minimum_bandwidth:
                model.addCons(minimum_bandwidth_vars[i] == link_vars[i], f"MinBandwidth_{i}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_flows': 3000,
        'n_links': 12,
        'min_bandwidth': 9,
        'max_bandwidth': 2185,
        'max_path_length': 18,
        'n_servers': 900,
        'server_min_capacity': 0,
        'server_max_capacity': 25,
    }
    cloud_allocation = CloudServerAllocation(parameters, seed=42)
    instance = cloud_allocation.get_instance()
    solve_status, solve_time = cloud_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")