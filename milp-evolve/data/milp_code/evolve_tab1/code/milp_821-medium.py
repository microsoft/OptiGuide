import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CyberPhysicalSecurityNetwork:
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
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_communication_requests(self, G):
        requests = []
        for k in range(self.n_requests):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            demand_k = int(np.random.uniform(*self.d_range))
            requests.append((o_k, d_k, demand_k))
        return requests

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_requests = np.random.randint(self.min_n_requests, self.max_n_requests + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        requests = self.generate_communication_requests(G)

        res = {
            'requests': requests,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'malicious_node_prob': self.malicious_node_prob,
            'max_zone_capacity': self.max_zone_capacity
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        requests = instance['requests']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        max_zone_capacity = instance['max_zone_capacity']

        model = Model("CyberPhysicalSecurityNetwork")

        NetworkSecurityArcUsage = {f"NSA_{i+1}_{j+1}_{k+1}": model.addVar(vtype="B", name=f"NSA_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_requests)}
        MaliciousNode = {f"MN_{i+1}": model.addVar(vtype="B", name=f"MN_{i+1}") for i in range(self.n_nodes)}
        CipherCapacity = {f"CC_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"CC_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_requests)}

        # Objective function
        objective_expr = quicksum(
            requests[k][2] * adj_mat[i, j][0] * CipherCapacity[f"CC_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_requests)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * NetworkSecurityArcUsage[f"NSA_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_requests)
        )
        
        # Minimize cost of detecting malicious nodes
        objective_expr += quicksum(
            self.malicious_detection_cost * MaliciousNode[f"MN_{i+1}"]
            for i in range(self.n_nodes)
        )

        model.setObjective(objective_expr, "minimize")

        # Flow conservation constraints
        for i in range(self.n_nodes):
            for k in range(self.n_requests):
                delta_i = 1 if requests[k][0] == i else -1 if requests[k][1] == i else 0
                flow_expr = quicksum(CipherCapacity[f"CC_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(CipherCapacity[f"CC_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Capacity constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(requests[k][2] * CipherCapacity[f"CC_{i+1}_{j+1}_{k+1}"] for k in range(self.n_requests))
            model.addCons(arc_expr <= adj_mat[i, j][2] * NetworkSecurityArcUsage[f"NSA_{i+1}_{j+1}_{k+1}"], f"arc_{i+1}_{j+1}")

        # Zonal capacity control
        for i in range(self.n_nodes):
            zone_cap_expr = quicksum(CipherCapacity[f"CC_{i+1}_{j+1}_{k+1}"] for j in outcommings[i] for k in range(self.n_requests)) \
                    + quicksum(CipherCapacity[f"CC_{j+1}_{i+1}_{k+1}"] for j in incommings[i] for k in range(self.n_requests))
            model.addCons(zone_cap_expr <= max_zone_capacity, f"zone_{i+1}")

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
        'c_range': (11, 50),
        'd_range': (100, 1000),
        'ratio': 900,
        'k_max': 90,
        'er_prob': 0.45,
        'malicious_detection_cost': 750,
        'malicious_node_prob': 0.05,
        'max_zone_capacity': 5000,
    }

    cyber_physical_sec_net = CyberPhysicalSecurityNetwork(parameters, seed=seed)
    instance = cyber_physical_sec_net.generate_instance()
    solve_status, solve_time = cyber_physical_sec_net.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")