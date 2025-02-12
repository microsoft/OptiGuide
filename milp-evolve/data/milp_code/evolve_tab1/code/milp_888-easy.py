import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NetworkFlowOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def path_delays(self):
        base_delay = 5.0  # base delay time in units
        return base_delay * np.random.rand(self.n_packets, self.n_paths)
    
    def generate_instance(self):
        packet_rates = self.randint(self.n_packets, self.packet_rate_interval)
        path_capacities = self.randint(self.n_paths, self.path_capacity_interval)
        connection_costs = self.randint(self.n_paths, self.connection_cost_interval)
        path_delays = self.path_delays()

        path_capacities = path_capacities * self.ratio * np.sum(packet_rates) / np.sum(path_capacities)
        path_capacities = np.round(path_capacities)

        packet_priorities = self.randint(self.n_packets, (0, 2))  # Binary {0,1} priority

        res = {
            'packet_rates': packet_rates,
            'path_capacities': path_capacities,
            'connection_costs': connection_costs,
            'path_delays': path_delays,
            'packet_priorities': packet_priorities
        }

        path_packet_combination = np.random.binomial(1, 0.5, (self.n_packets, self.n_paths, self.hyper_param))  # New combinatorial data element

        res['path_packet_combination'] = path_packet_combination

        return res

    def solve(self, instance):
        # Instance data
        packet_rates = instance['packet_rates']
        path_capacities = instance['path_capacities']
        connection_costs = instance['connection_costs']
        path_delays = instance['path_delays']
        packet_priorities = instance['packet_priorities']
        path_packet_combination = instance['path_packet_combination']

        n_packets = len(packet_rates)
        n_paths = len(path_capacities)

        model = Model("NetworkFlowOptimization")

        # Decision variables
        activate_path = {j: model.addVar(vtype="B", name=f"Activate_{j}") for j in range(n_paths)}
        route_packet = {(i, j): model.addVar(vtype="B", name=f"Route_{i}_{j}") for i in range(n_packets) for j in range(n_paths)}
        highest_utilization_activate = {j: model.addVar(vtype="B", name=f"HighestUtilizationActivate_{j}") for j in range(n_paths)}
        
        # New variables for convex hull
        path_packet_combination_var = {(i, j, k): model.addVar(vtype="B", name=f"Comb_{i}_{j}_{k}") for i in range(n_packets) for j in range(n_paths) for k in range(self.hyper_param)}

        # Objective: Maximize throughput, minimize the total cost including activation and penalty for exceeding capacity
        penalty_per_exceed_capacity = 100
        highest_utilization_penalty = 200  # Extra penalty for high-utilization paths
        objective_expr = quicksum(-connection_costs[j] * activate_path[j] for j in range(n_paths)) - penalty_per_exceed_capacity * quicksum(path_delays[i, j] * route_packet[i, j] for i in range(n_packets) for j in range(n_paths)) - quicksum(highest_utilization_penalty * highest_utilization_activate[j] for j in range(n_paths) if any(packet_priorities[i] for i in range(n_packets)))

        # Constraints: each packet must be sent through exactly one path
        for i in range(n_packets):
            model.addCons(quicksum(route_packet[i, j] for j in range(n_paths)) == 1, f"Packet_Routing_{i}")
        
        # Constraints: path capacity limits must be respected
        for j in range(n_paths):
            model.addCons(quicksum(packet_rates[i] * route_packet[i, j] for i in range(n_packets)) <= path_capacities[j] * activate_path[j], f"Path_Capacity_{j}")

        # Constraint: Average delay minimized (All packets routed within permissible limits)
        for i in range(n_packets):
            for j in range(n_paths):
                model.addCons(path_delays[i, j] * route_packet[i, j] <= activate_path[j] * 50, f"Delay_Limit_{i}_{j}")

        # New Constraints:
        for i in range(n_packets):
            if packet_priorities[i]:  # For high-priority packets
                model.addCons(quicksum(route_packet[i, j] for j in range(n_paths) if highest_utilization_activate[j]) >= 1, f"HighUtilizationCover_{i}")
                for j in range(n_paths):
                    model.addCons(highest_utilization_activate[j] <= activate_path[j], f"HighUtilizationLink_{j}")

        # Convex Hull Constraints
        for i in range(n_packets):
            for j in range(n_paths):
                for k in range(self.hyper_param):
                    model.addCons(path_packet_combination_var[i, j, k] <= path_packet_combination[i, j, k], f"Comb_Hull_{i}_{j}_{k}")
                    model.addCons(quicksum(path_packet_combination_var[i, j, k] for k in range(self.hyper_param)) == route_packet[i, j], f"Comb_Hull_Route_{i}_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_packets': 50,
        'n_paths': 25,
        'packet_rate_interval': (14, 140),
        'path_capacity_interval': (450, 1350),
        'connection_cost_interval': (600, 1200),
        'ratio': 90.0,
        'hyper_param': 10,
    }

    network_optimization = NetworkFlowOptimization(parameters, seed=seed)
    instance = network_optimization.generate_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")