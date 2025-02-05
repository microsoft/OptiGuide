import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FCMCNFSetCover:
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
    
    def generate_node_activation_costs(self):
        node_activation_costs = np.random.uniform(self.min_node_activation_cost, self.max_node_activation_cost, self.n_nodes)
        return node_activation_costs

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        node_activation_costs = self.generate_node_activation_costs()

        return {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'node_activation_costs': node_activation_costs
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        node_activation_costs = instance['node_activation_costs']

        model = Model("FCMCNFSetCover")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        # New node activation variables
        z_vars = {f"z_{i+1}": model.addVar(vtype="B", name=f"z_{i+1}") for i in range(self.n_nodes)}

        # Original objective for edge costs
        edge_cost_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        edge_cost_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        # New objective for node activation costs
        node_activation_cost_expr = quicksum(
            node_activation_costs[i] * z_vars[f"z_{i+1}"]
            for i in range(self.n_nodes)
        )

        objective_expr = edge_cost_expr + node_activation_cost_expr

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # New constraint to ensure that flow passes through only activated nodes
        for (i, j) in edge_list:
            for k in range(self.n_commodities):
                model.addCons(x_vars[f"x_{i+1}_{j+1}_{k+1}"] <= self.big_M * z_vars[f"z_{i+1}"], f"node_activation_{i+1}_{k+1}")
                model.addCons(x_vars[f"x_{i+1}_{j+1}_{k+1}"] <= self.big_M * z_vars[f"z_{j+1}"], f"node_activation_{j+1}_{k+1}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 60,
        'max_n_nodes': 90,
        'min_n_commodities': 7,
        'max_n_commodities': 22,
        'c_range': (110, 500),
        'd_range': (30, 300),
        'ratio': 50,
        'k_max': 50,
        'er_prob': 0.31,
        'min_node_activation_cost': 10,
        'max_node_activation_cost': 250,
        'big_M': 10000,
    }

    fcmcnf_set_cover = FCMCNFSetCover(parameters, seed=seed)
    instance = fcmcnf_set_cover.generate_instance()
    solve_status, solve_time = fcmcnf_set_cover.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")