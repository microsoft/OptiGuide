import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class DataCenterTrafficOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def generate_instance(self):
        graph = nx.barabasi_albert_graph(self.num_nodes, self.num_edges)
        customer_groups = list(graph.nodes)[:self.num_customer_groups]
        servers = list(graph.nodes)[:self.num_servers]
        
        bandwidths = self.randint(self.num_customer_groups, self.bandwidth_interval)
        capacities = self.randint(self.num_servers, self.capacity_interval)
        fixed_costs = self.randint(self.num_servers, self.fixed_cost_interval)
        operational_costs = self.randint(self.num_servers, self.operational_cost_interval)

        connection_weights = {(u, v): np.random.rand()*10 for u, v in graph.edges}
        
        res = {
            'bandwidths': bandwidths,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'operational_costs': operational_costs,
            'graph': graph,
            'connection_weights': connection_weights,
            'customer_groups': customer_groups,
            'servers': servers
        }

        reliability_coeff = 100 * np.random.rand()
        res.update({"reliability_coeff": reliability_coeff})

        return res

    def solve(self, instance):
        bandwidths = instance['bandwidths']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        operational_costs = instance['operational_costs']
        graph = instance['graph']
        connection_weights = instance['connection_weights']
        servers = instance['servers']
        customer_groups = instance['customer_groups']
        reliability_coeff = instance['reliability_coeff']

        num_customer_groups = len(customer_groups)
        num_servers = len(servers)
        
        model = Model("DataCenterTrafficOptimization")
        
        server_operation = {j: model.addVar(vtype="B", name=f"ServerOperation_{j}") for j in servers}
        network_traffic = {(i, j): model.addVar(vtype="B", name=f"NetworkTraffic_{i}_{j}") for i in customer_groups for j in servers}
        reliability = {j: model.addVar(vtype="I", name=f"Reliability_{j}") for j in servers}
        high_bandwidth_cost = {j: model.addVar(vtype="C", name=f"HighBandwidthCost_{j}") for j in servers}

        # Objective: Minimize latency and operational costs, maximizing reliability
        objective_expr = quicksum(fixed_costs[j] * server_operation[j] for j in servers) + \
                         quicksum(connection_weights[u, v] * network_traffic[i, j] for i in customer_groups for j in servers
                                  for u, v in graph.edges if u == i and v == j) + \
                         quicksum(reliability_coeff * reliability[j] for j in servers) + \
                         quicksum(operational_costs[j] * high_bandwidth_cost[j] for j in servers)

        model.setObjective(objective_expr, "minimize")

        # Constraint: Each customer group must be connected to at least one server
        for i in customer_groups:
            model.addCons(quicksum(network_traffic[i, j] for j in servers) >= 1, f"CustomerGroup_{i}")
        
        # Constraint: Server capacity constraints
        for j in servers:
            model.addCons(quicksum(network_traffic[i, j] * bandwidths[i] for i in customer_groups) <= capacities[j] * server_operation[j], f"MaximumCapacity_{j}")

        # Constraint: Total operational cost constraints
        total_cost = np.sum(fixed_costs)
        model.addCons(quicksum(fixed_costs[j] * server_operation[j] for j in servers) <= total_cost, "MaximumCost")

        # Constraint: Assigning high bandwidth costs for servers with high operational load
        high_cost_threshold = np.percentile(fixed_costs, 80)
        for j in servers:
            if fixed_costs[j] > high_cost_threshold:
                model.addCons(high_bandwidth_cost[j] >= fixed_costs[j], f"HighBandwidthCost_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_nodes': 350,
        'num_edges': 27,
        'num_customer_groups': 225,
        'num_servers': 80,
        'bandwidth_interval': (35, 140),
        'capacity_interval': (500, 2000),
        'fixed_cost_interval': (100, 500),
        'operational_cost_interval': (7, 70),
    }

    data_center_optimization = DataCenterTrafficOptimization(parameters, seed=seed)
    instance = data_center_optimization.generate_instance()
    solve_status, solve_time = data_center_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")