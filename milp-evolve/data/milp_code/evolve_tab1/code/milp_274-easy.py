import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NetworkTrafficOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_network_latency(self):
        return np.random.rand(self.n_clients, self.n_nodes) * self.latency_scale

    def bandwidth_supply(self):
        return np.random.rand(self.n_bandwidths) * self.bandwidth_capacity_scale

    def generate_instance(self):
        requests = self.randint(self.n_clients, self.request_interval)
        node_capacities = self.randint(self.n_nodes, self.node_capacity_interval)
        bandwidth_capacities = self.bandwidth_supply()
        fixed_costs = self.randint(self.n_nodes, self.fixed_cost_interval)
        latency_costs = self.unit_network_latency()
        
        # Generate a subset of traffic sources requiring mandatory handling
        mandatory_sources = random.sample(range(self.number_of_requests), self.number_of_mandatory_requests)

        res = {
            'requests': requests,
            'node_capacities': node_capacities,
            'bandwidth_capacities': bandwidth_capacities,
            'fixed_costs': fixed_costs,
            'latency_costs': latency_costs,
            'mandatory_sources': mandatory_sources,
            'request_profits': self.randint(self.number_of_requests, (10, 100)),
            'request_sizes': self.randint(self.number_of_requests, (1, 10))
        }
        
        return res

    def solve(self, instance):
        requests = instance['requests']
        node_capacities = instance['node_capacities']
        bandwidth_capacities = instance['bandwidth_capacities']
        fixed_costs = instance['fixed_costs']
        latency_costs = instance['latency_costs']
        mandatory_sources = instance['mandatory_sources']
        request_profits = instance['request_profits']
        request_sizes = instance['request_sizes']

        n_clients, n_nodes, n_bandwidths = len(requests), len(node_capacities), len(bandwidth_capacities)
        number_of_requests = len(request_profits)

        model = Model("NetworkTrafficOptimization")

        node_open = {j: model.addVar(vtype="B", name=f"NodeOpen_{j}") for j in range(n_nodes)}
        client_requests = {(i, j): model.addVar(vtype="C", name=f"ClientRequests_{i}_{j}") for i in range(n_clients) for j in range(n_nodes)}
        bandwidth = {j: model.addVar(vtype="C", name=f"Bandwidth_{j}") for j in range(n_bandwidths)}
        request_handling = {(i, j): model.addVar(vtype="B", name=f"Handling_{i}_{j}") for i in range(number_of_requests) for j in range(n_nodes)}

        objective_expr = quicksum(fixed_costs[j] * node_open[j] for j in range(n_nodes)) + \
                         quicksum(latency_costs[i, j] * client_requests[i, j] for i in range(n_clients) for j in range(n_nodes)) + \
                         quicksum(request_profits[i] * request_handling[(i, j)] for i in range(number_of_requests) for j in range(n_nodes))

        model.setObjective(objective_expr, "minimize")

        for i in range(n_clients):
            model.addCons(quicksum(client_requests[i, j] for j in range(n_nodes)) == requests[i], f"ClientRequest_{i}")

        for j in range(n_nodes):
            model.addCons(quicksum(client_requests[i, j] for i in range(n_clients)) <= node_capacities[j] * node_open[j], f"NodeCapacity_{j}")

        for k in range(n_bandwidths):
            model.addCons(bandwidth[k] <= bandwidth_capacities[k], f"BandwidthCapacity_{k}")

        for j in range(n_nodes):
            model.addCons(quicksum(bandwidth[k] for k in range(n_bandwidths)) >= quicksum(client_requests[i, j] for i in range(n_clients)) * node_open[j], f"BandwidthSupplyLink_{j}")

        for i in range(number_of_requests):
            model.addCons(quicksum(request_handling[(i, j)] for j in range(n_nodes)) <= 1, f"RequestAssignment_{i}")

        for j in range(n_nodes):
            model.addCons(quicksum(request_sizes[i] * request_handling[(i, j)] for i in range(number_of_requests)) <= node_capacities[j], f"HandlingCapacity_{j}")

        for m_src in mandatory_sources:
            model.addCons(quicksum(request_handling[(m_src, j)] for j in range(n_nodes)) >= 1, f"MandatorySourceCover_{m_src}")

        specific_node, specific_request = 0, 2
        model.addCons(node_open[specific_node] == request_handling[(specific_request, specific_node)], f"LogicalCondition_RequestPlacement_{specific_request}_{specific_node}")

        for j in range(n_nodes):
            model.addCons(quicksum(client_requests[i, j] for i in range(n_clients)) * node_open[j] <= quicksum(bandwidth[k] for k in range(n_bandwidths)), f"LogicalCondition_NodeBandwidth_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_clients': 25,
        'n_nodes': 600,
        'n_bandwidths': 30,
        'request_interval': (560, 2800),
        'node_capacity_interval': (525, 2100),
        'bandwidth_capacity_scale': 750.0,
        'fixed_cost_interval': (131, 525),
        'latency_scale': 810.0,
        'number_of_requests': 200,
        'number_of_mandatory_requests': 10,
    }

    network_optimization = NetworkTrafficOptimization(parameters, seed=seed)
    instance = network_optimization.generate_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")