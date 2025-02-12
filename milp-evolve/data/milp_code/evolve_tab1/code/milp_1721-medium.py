import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FCMCNF:
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

    def generate_commodities(self, G):
        commodities = []
        for k in range(self.n_commodities):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        # Generate data for facility location and flow threshold
        self.n_facilities = self.n_nodes
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        flow_thresholds = np.random.randint(self.min_flow_thr, self.max_flow_thr + 1, self.n_facilities)
        
        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'facility_costs': facility_costs,
            'flow_thresholds': flow_thresholds
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        facility_costs = instance['facility_costs']
        flow_thresholds = instance['flow_thresholds']

        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        open_vars = {i: model.addVar(vtype="B", name=f"Facility_{i+1}") for i in range(self.n_nodes)}

        # Objective
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr -= quicksum(
            facility_costs[i] * open_vars[i]
            for i in range(self.n_nodes)
        )

        # Flow Conservation
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Arc Capacity Constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # Facility Opening based on Big M Method
        for i in range(self.n_nodes):
            flow_threshold = flow_thresholds[i]
            total_flow = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i] for k in range(self.n_commodities)) + \
                         quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i] for k in range(self.n_commodities))
            big_M = sum(commodities[k][2] for k in range(self.n_commodities))
            model.addCons(total_flow >= flow_threshold - (1 - open_vars[i]) * big_M, f"flow_threshold_{i+1}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 11,
        'max_n_nodes': 15,
        'min_n_commodities': 44,
        'max_n_commodities': 495,
        'c_range': (3, 18),
        'd_range': (250, 2500),
        'ratio': 1800,
        'k_max': 150,
        'er_prob': 0.38,
        'min_facility_cost': 200,
        'max_facility_cost': 250,
        'min_flow_thr': 300,
        'max_flow_thr': 1500,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")