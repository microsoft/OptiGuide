'''
Simplified Fixed Charge Multicommodity Network Flow (SFCMCNF)
'''
import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SFCMCNF:
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
            # integer demands
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        model = Model("SFCMCNF")
        # Variables representing the total flow on each edge
        total_flow_vars = {f"tf_{i+1}_{j+1}": model.addVar(vtype="C", name=f"tf_{i+1}_{j+1}") for (i, j) in edge_list}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}

        # Objective: Minimize transportation and fixed edge costs
        objective_expr = quicksum(
            adj_mat[i, j][0] * total_flow_vars[f"tf_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        # Flow conservation constraint per node
        for i in range(self.n_nodes):
            incoming_flow = quicksum(total_flow_vars[f"tf_{j+1}_{i+1}"] for j in incommings[i])
            outgoing_flow = quicksum(total_flow_vars[f"tf_{i+1}_{j+1}"] for j in outcommings[i])
            # Source nodes supply their demand, sink nodes consume their demand
            supply = 0
            demand = 0
            for o, d, d_k in commodities:
                if o == i:
                    supply += d_k
                if d == i:
                    demand += d_k

            model.addCons(incoming_flow - outgoing_flow == demand - supply, f"node_{i+1}")

        for (i, j) in edge_list:
            model.addCons(total_flow_vars[f"tf_{i+1}_{j+1}"] <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"capacity_{i+1}_{j+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 20,
        'max_n_nodes': 30,
        'min_n_commodities': 30,
        'max_n_commodities': 45, 
        'c_range': (11, 50),
        'd_range': (10, 100),
        'ratio': 100,
        'k_max': 10,
        'er_prob': 0.3,
    }

    sfcfcnf = SFCMCNF(parameters, seed=seed)
    instance = sfcfcnf.generate_instance()
    solve_status, solve_time = sfcfcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")