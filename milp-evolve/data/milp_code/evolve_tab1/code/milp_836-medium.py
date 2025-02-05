import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HealthLogisticsNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            logistics_cost = np.random.uniform(*self.logistics_cost_range)
            emergency_penalty = np.random.uniform(self.logistics_cost_range[0] * self.ratio, self.logistics_cost_range[1] * self.ratio)
            vehicle_capacity = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.supply_demand_range)
            adj_mat[i, j] = (logistics_cost, emergency_penalty, vehicle_capacity)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_supply_requests(self, G):
        requests = []
        for k in range(self.n_requests):
            while True:
                source = np.random.randint(0, self.n_nodes)
                destination = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, source, destination) and source != destination:
                    break
            demand_k = int(np.random.uniform(*self.supply_demand_range))
            emergency_status = np.random.choice([0, 1], p=[1-self.emergency_prob, self.emergency_prob])
            requests.append((source, destination, demand_k, emergency_status))
        return requests

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_requests = np.random.randint(self.min_n_requests, self.max_n_requests + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        requests = self.generate_supply_requests(G)

        res = {
            'requests': requests,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'max_hospital_capacity': self.max_hospital_capacity
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        requests = instance['requests']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        max_hospital_capacity = instance['max_hospital_capacity']

        model = Model("HealthLogisticsNetwork")

        NHV = {f"NHV_{i+1}_{j+1}_{k+1}": model.addVar(vtype="B", name=f"NHV_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_requests)}
        EV = {f"EV_{i+1}": model.addVar(vtype="B", name=f"EV_{i+1}") for i in range(self.n_nodes)}
        SFC = {f"SFC_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"SFC_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_requests)}

        # Objective function
        objective_expr = quicksum(
            requests[k][2] * adj_mat[i, j][0] * SFC[f"SFC_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_requests)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * NHV[f"NHV_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_requests)
        )
        
        # Minimize penalty of not prioritizing emergency deliveries
        objective_expr += quicksum(
            self.emergency_penalty * EV[f"EV_{i+1}"]
            for i in range(self.n_nodes)
        )

        model.setObjective(objective_expr, "minimize")

        # Supply flow conservation constraints
        for i in range(self.n_nodes):
            for k in range(self.n_requests):
                delta_i = 1 if requests[k][0] == i else -1 if requests[k][1] == i else 0
                flow_expr = quicksum(SFC[f"SFC_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(SFC[f"SFC_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Vehicle capacity constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(requests[k][2] * SFC[f"SFC_{i+1}_{j+1}_{k+1}"] for k in range(self.n_requests))
            model.addCons(arc_expr <= adj_mat[i, j][2] * NHV[f"NHV_{i+1}_{j+1}_{k+1}"], f"arc_{i+1}_{j+1}")

        # Hospital capacity control
        for i in range(self.n_nodes):
            hospital_cap_expr = quicksum(SFC[f"SFC_{i+1}_{j+1}_{k+1}"] for j in outcommings[i] for k in range(self.n_requests)) \
                    + quicksum(SFC[f"SFC_{j+1}_{i+1}_{k+1}"] for j in incommings[i] for k in range(self.n_requests))
            model.addCons(hospital_cap_expr <= max_hospital_capacity, f"hospital_{i+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
            
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 10,
        'max_n_nodes': 30,
        'min_n_requests': 300,
        'max_n_requests': 405,
        'logistics_cost_range': (3, 20),
        'supply_demand_range': (50, 300),
        'ratio': 800,
        'k_max': 50,
        'er_prob': 0.30,
        'emergency_penalty': 1000,
        'emergency_prob': 0.1,
        'max_hospital_capacity': 1200,
    }

    health_logistics_net = HealthLogisticsNetwork(parameters, seed=seed)
    instance = health_logistics_net.generate_instance()
    solve_status, solve_time = health_logistics_net.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")