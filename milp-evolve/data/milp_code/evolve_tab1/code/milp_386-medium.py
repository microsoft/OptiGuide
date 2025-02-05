import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_rows * self.n_cols * self.density)
        indices = np.random.choice(self.n_cols, size=nnzrs)  
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  

        _, col_nrows = np.unique(indices, return_counts=True)
        indices[:self.n_rows] = np.random.permutation(self.n_rows) 
        i = 0
        indptr = [0]
        for n in col_nrows:
            if i >= self.n_rows:
                indices[i:i + n] = np.random.choice(self.n_rows, size=n, replace=False)
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i + n] = np.random.choice(remaining_rows, size=i + n - self.n_rows, replace=False)
            i += n
            indptr.append(i)

        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        graph = nx.barabasi_albert_graph(self.n_cols, self.num_edges_per_node, seed=self.seed)
        capacities = np.random.randint(1, self.max_capacity, size=len(graph.edges))
        flows = np.random.uniform(0, self.max_flow, size=len(graph.edges))

        source_node, sink_node = 0, self.n_cols - 1

        adj_list = {i: [] for i in range(self.n_cols)}
        for idx, (u, v) in enumerate(graph.edges):
            adj_list[u].append((v, flows[idx], capacities[idx]))
            adj_list[v].append((u, flows[idx], capacities[idx]))  

        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {'c': c,
               'indptr_csr': indptr_csr,
               'indices_csr': indices_csr,
               'adj_list': adj_list,
               'source_node': source_node,
               'sink_node': sink_node}

        time_windows = {i: (np.random.randint(0, self.latest_delivery_time // 2), 
                            np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) 
                        for i in range(self.n_cols)}
        travel_times = {(u, v): np.random.randint(1, self.max_travel_time) for u, v in graph.edges}
        uncertainty = {i: np.random.normal(0, self.time_uncertainty_stddev, size=2) for i in range(self.n_cols)}

        res.update({'time_windows': time_windows, 'travel_times': travel_times, 'uncertainty': uncertainty})

        Big_M_values = np.random.randint(100, 500, size=self.n_cols)
        res.update({'Big_M_values': Big_M_values}) 

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        adj_list = instance['adj_list']
        source_node = instance['source_node']
        sink_node = instance['sink_node']
        time_windows = instance['time_windows']
        uncertainty = instance['uncertainty']
        Big_M_values = instance['Big_M_values']

        model = Model("SimplifiedLogisticsOptimization")
        var_names = {}

        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        flow_vars = {}
        for u in adj_list:
            for v, _, capacity in adj_list[u]:
                flow_vars[(u, v)] = model.addVar(vtype='C', lb=0, ub=capacity, name=f"f_{u}_{v}")

        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        for node in adj_list:
            if node == source_node:
                model.addCons(quicksum(flow_vars[(source_node, v)] for v, _, _ in adj_list[source_node]) >= 1,
                              f"flow_source_{source_node}")
            elif node == sink_node:
                model.addCons(quicksum(flow_vars[(u, sink_node)] for u, _, _ in adj_list[sink_node]) >= 1,
                              f"flow_sink_{sink_node}")
            else:
                inflow = quicksum(flow_vars[(u, node)] for u, _, _ in adj_list[node] if (u, node) in flow_vars)
                outflow = quicksum(flow_vars[(node, v)] for v, _, _ in adj_list[node] if (node, v) in flow_vars)
                model.addCons(inflow - outflow == 0, f"flow_balance_{node}")

        time_vars = {}
        early_penalty_vars = {}
        late_penalty_vars = {}
        activation_vars = {}

        for j in range(self.n_cols):
            time_vars[j] = model.addVar(vtype='C', name=f"t_{j}")
            early_penalty_vars[j] = model.addVar(vtype='C', name=f"e_{j}")
            late_penalty_vars[j] = model.addVar(vtype='C', name=f"l_{j}")
            activation_vars[j] = model.addVar(vtype='B', name=f"activation_{j}")

            start_window, end_window = time_windows[j]
            uncertainty_start, uncertainty_end = uncertainty[j]
            Big_M = Big_M_values[j]

            model.addCons(time_vars[j] >= (start_window + uncertainty_start) - Big_M * (1 - activation_vars[j]), 
                          f"time_window_start_M_{j}")
            model.addCons(time_vars[j] <= (end_window + uncertainty_end) + Big_M * (1 - activation_vars[j]), 
                          f"time_window_end_M_{j}")

            model.addCons(early_penalty_vars[j] >= (start_window + uncertainty_start) - time_vars[j], 
                          f"early_penalty_{j}")
            model.addCons(late_penalty_vars[j] >= time_vars[j] - (end_window + uncertainty_end), 
                          f"late_penalty_{j}")

        cost_term = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        flow_term = quicksum(flow_vars[(u, v)] for u, v in flow_vars)
        time_penalty_term = quicksum(early_penalty_vars[j] + late_penalty_vars[j] for j in range(self.n_cols))

        objective_expr = cost_term - self.flow_weight * flow_term + self.time_penalty_weight * time_penalty_term

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 3000,
        'n_cols': 140,
        'density': 0.17,
        'max_coef': 2250,
        'num_edges_per_node': 15,
        'max_capacity': 1200,
        'max_flow': 25,
        'flow_weight': 0.73,
        'latest_delivery_time': 960,
        'max_travel_time': 60,
        'time_penalty_weight': 0.66,
        'time_uncertainty_stddev': 30,
    }

    simplified_logistics_optimization = SimplifiedLogisticsOptimization(parameters, seed=seed)
    instance = simplified_logistics_optimization.generate_instance()
    solve_status, solve_time = simplified_logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")