import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SetCoverWithNetwork:
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
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        crucial_sets = np.random.choice(self.n_cols, self.n_crucial, replace=False)
        activation_cost = np.random.randint(self.activation_cost_low, self.activation_cost_high, size=self.n_crucial)
        co2_emissions = np.random.rand(self.n_cols)
        co2_limit = np.random.rand() * self.co2_emission_factor

        G = nx.erdos_renyi_graph(n=self.network_nodes, p=self.network_prob, directed=True, seed=self.seed)
        for u, v in G.edges():
            G[u][v]['handling_time'] = np.random.randint(1, 10)
            G[u][v]['congestion_cost'] = np.random.randint(1, 10)      # Added congestion cost
            G[u][v]['noise_cost'] = np.random.randint(1, 5)            # Added noise cost
        
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)

        res = {
            'c': c,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'crucial_sets': crucial_sets,
            'activation_cost': activation_cost,
            'co2_emissions': co2_emissions,
            'co2_limit': co2_limit,
            'graph': G,
            'invalid_edges': E_invalid
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        crucial_sets = instance['crucial_sets']
        activation_cost = instance['activation_cost']
        co2_emissions = instance['co2_emissions']
        co2_limit = instance['co2_limit']
        G = instance['graph']
        E_invalid = instance['invalid_edges']

        model = Model("SetCoverWithNetwork")
        var_names = {}
        activate_crucial = {}
        handle_cost_vars = {}
        congestion_vars = {}

        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        for idx, j in enumerate(crucial_sets):
            activate_crucial[j] = model.addVar(vtype="B", name=f"y_{j}", obj=activation_cost[idx])
        
        for j in range(self.n_cols):
            for u, v in G.edges():
                handle_cost_vars[(j, u, v)] = model.addVar(vtype="B", name=f"handle_cost_{j}_{u}_{v}")
                congestion_vars[(j, u, v)] = model.addVar(vtype="B", name=f"congestion_{j}_{u}_{v}")

        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        for j in crucial_sets:
            rows_impacting_j = np.where(indices_csr == j)[0]
            for row in rows_impacting_j:
                model.addCons(var_names[j] >= activate_crucial[j], f"crucial_coverage_row_{row}_set_{j}")

        model.addCons(quicksum(var_names[j] * co2_emissions[j] for j in range(self.n_cols)) <= co2_limit, "co2_limit")

        for j in range(self.n_cols):
            for u, v in G.edges():
                if (u, v) in E_invalid:
                    model.addCons(handle_cost_vars[(j, u, v)] == 0, f"InvalidEdge_{j}_{u}_{v}")
                else:
                    model.addCons(handle_cost_vars[(j, u, v)] <= 1, f"ValidEdge_{j}_{u}_{v}")

        # Congestion constraint
        for j in range(self.n_cols):
            total_handling = quicksum(handle_cost_vars[(j, u, v)] for u, v in G.edges())
            model.addCons(total_handling <= self.max_handling, f"HandlingLimit_{j}")
        
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + \
                         quicksum(activate_crucial[j] * activation_cost[idx] for idx, j in enumerate(crucial_sets)) - \
                         quicksum(G[u][v]['handling_time'] * handle_cost_vars[(j, u, v)] for j in range(self.n_cols) for u, v in G.edges()) - \
                         quicksum(G[u][v]['congestion_cost'] * congestion_vars[(j, u, v)] for j in range(self.n_cols) for u, v in G.edges()) - \
                         quicksum(G[u][v]['noise_cost'] * congestion_vars[(j, u, v)] for j in range(self.n_cols) for u, v in G.edges())

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 112,
        'n_cols': 1125,
        'density': 0.38,
        'max_coef': 7,
        'n_crucial': 630,
        'activation_cost_low': 157,
        'activation_cost_high': 375,
        'co2_emission_factor': 1875.0,
        'network_nodes': 7,
        'network_prob': 0.24,
        'exclusion_rate': 0.8,
        'max_handling': 100,
    }

    set_cover_network_problem = SetCoverWithNetwork(parameters, seed=seed)
    instance = set_cover_network_problem.generate_instance()
    solve_status, solve_time = set_cover_network_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")