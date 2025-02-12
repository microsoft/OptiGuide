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

    def generate_barabasi_graph(self):
        G = nx.barabasi_albert_graph(n=self.n_nodes, m=2, seed=self.seed)
        G = nx.DiGraph([(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()])
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
    
    def generate_traffic_data(self):
        traffic_patterns = {i: np.random.choice([1.0, 1.5, 2.0], size=self.n_time_periods, p=[0.5, 0.3, 0.2]) for i in range(self.n_nodes)}
        return traffic_patterns
    
    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        if random.choice([True, False]):
            G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        else:
            G, adj_mat, edge_list, incommings, outcommings = self.generate_barabasi_graph()
        commodities = self.generate_commodities(G)
        traffic_patterns = self.generate_traffic_data()
        
        res = {
            'commodities': commodities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'traffic_patterns': traffic_patterns
        }

        ### given instance data code ends here
        flow_thresholds = {(i, j): adj_mat[i, j][0] * self.flow_threshold_factor for (i, j) in edge_list}
        res['flow_thresholds'] = flow_thresholds
        ### new instance data code ends here
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        traffic_patterns = instance['traffic_patterns']
        flow_thresholds = instance['flow_thresholds']

        model = Model("FCMCNF")

        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        z_vars = {f"z_{k+1}": model.addVar(vtype="I", lb=0, name=f"z_{k+1}") for k in range(self.n_commodities)}
        w_vars = {f"w_{i+1}_{j+1}": model.addVar(vtype="B", name=f"w_{i+1}_{j+1}") for (i, j) in edge_list}

        # Objective Function: Include penalties for unmet demand and expected traffic delays
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            z_vars[f"z_{k+1}"] * 100  # Penalty for unmet demand
            for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][2] * traffic_patterns[i][j % self.n_time_periods] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        # Flow Constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")
        
        # Capacity Constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")
        
        # Unmet Demand Constraints
        for k in range(self.n_commodities):
            demand_expr = quicksum(x_vars[f"x_{commodities[k][0]+1}_{j+1}_{k+1}"] for j in outcommings[commodities[k][0]]) - quicksum(x_vars[f"x_{j+1}_{commodities[k][0]+1}_{k+1}"] for j in incommings[commodities[k][0]])
            model.addCons(demand_expr + z_vars[f"z_{k+1}"] >= commodities[k][2], f"demand_{k+1}")
        
        # Logical Condition: If flow exceeds threshold, w_var must be 1
        for (i, j) in edge_list:
            for k in range(self.n_commodities):
                model.addCons(x_vars[f"x_{i+1}_{j+1}_{k+1}"] >= flow_thresholds[(i, j)] * w_vars[f"w_{i+1}_{j+1}"], f"logic_flow_thresh_{i+1}_{j+1}_{k+1}")
                model.addCons(y_vars[f"y_{i+1}_{j+1}"] >= w_vars[f"w_{i+1}_{j+1}"], f"logic_y_flow_{i+1}_{j+1}_{k+1}")

        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 10,
        'max_n_nodes': 11,
        'min_n_commodities': 300,
        'max_n_commodities': 630,
        'c_range': (110, 500),
        'd_range': (270, 2700),
        'ratio': 3000,
        'k_max': 112,
        'er_prob': 0.8,
        'n_time_periods': 15,
    }
    flow_threshold_factor = 1.5
    parameters['flow_threshold_factor'] = flow_threshold_factor
    ### given parameter code ends here
    ### new parameter code ends here

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")