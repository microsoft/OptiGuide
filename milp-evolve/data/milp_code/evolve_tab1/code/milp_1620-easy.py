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
            adj_mat[i, j] = {'cost': c_ij, 'fixed': f_ij, 'capacity': u_ij}
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
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        piecewise_segments = self.generate_piecewise_segments(edge_list)

        res = {
            'commodities': commodities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'piecewise_segments': piecewise_segments
        }
        
        return res
    
    def generate_piecewise_segments(self, edge_list):
        segments = {}
        for (i, j) in edge_list:
            bp1 = np.random.uniform(1, 10)
            bp2 = np.random.uniform(10, 20)
            segments[(i, j)] = {'breakpoints': [0, bp1, bp2], 'slopes': [1, 2, 4]}  # Example: linear segments with different slopes
        return segments

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        piecewise_segments = instance['piecewise_segments']
        
        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        
        pw_vars = {}  # Piecewise segment variables
        for (i, j), segments in piecewise_segments.items():
            for m, bp in enumerate(segments['breakpoints'][1:], 1):  # excluding the first zero breakpoint
                pw_vars[(i, j, m)] = model.addVar(vtype="C", name=f"pw_{i+1}_{j+1}_{m}")

        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j]['cost'] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j]['fixed'] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            piecewise_segments[(i, j)]['slopes'][m-1] * pw_vars[(i, j, m)]
            for (i, j) in edge_list for m in range(1, len(piecewise_segments[(i, j)]['breakpoints']))
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j]['capacity'] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

            # Piecewise constraints: Total flow on arc (i, j) should match sum of piecewise vars
            model.addCons(arc_expr == quicksum(pw_vars[(i, j, m)] for m in range(1, len(piecewise_segments[(i, j)]['breakpoints']))), f"pw_total_{i+1}_{j+1}")
            
            # Ensure that piecewise vars follow breakpoints
            for m in range(1, len(piecewise_segments[(i, j)]['breakpoints'])):
                if m > 1:
                    model.addCons(pw_vars[(i, j, m)] >= 0, f"pw_nonnegative_{i+1}_{j+1}_{m}")
                    model.addCons(pw_vars[(i, j, m)] <= piecewise_segments[(i, j)]['breakpoints'][m] - piecewise_segments[(i, j)]['breakpoints'][m-1],
                                  f"pw_range_{i+1}_{j+1}_{m}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 20,
        'max_n_nodes': 22,
        'min_n_commodities': 15,
        'max_n_commodities': 315,
        'c_range': (22, 100),
        'd_range': (10, 100),
        'ratio': 100,
        'k_max': 50,
        'er_prob': 0.24,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")