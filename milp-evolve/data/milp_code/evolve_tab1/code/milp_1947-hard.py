import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexCoverage:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

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

        resource_consumption = np.random.randint(low=1, high=10, size=self.n_cols)
        resource_budget = int(0.6 * self.n_cols * 5)
        
        G = nx.barabasi_albert_graph(self.n_nodes, self.m_parameter)
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.randint(1, 10)
        
        res = {
            'c': c,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'resource_consumption': resource_consumption,
            'resource_budget': resource_budget,
            'graph': G,
        }
        return res

    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        resource_consumption = instance['resource_consumption']
        resource_budget = instance['resource_budget']
        graph = instance['graph']

        model = Model("ComplexCoverage")
        x_vars = {}
        y_vars = {}

        for j in range(self.n_cols):
            x_vars[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])
            y_vars[j] = model.addVar(vtype="I", name=f"y_{j}", lb=0, ub=1)

        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(x_vars[j] for j in cols) >= 1, f"c_{row}")

        total_resource_consumption = quicksum(resource_consumption[j] * x_vars[j] for j in range(self.n_cols))
        model.addCons(total_resource_consumption <= resource_budget, "resource_constraint")

        for u, v, data in graph.edges(data=True):
            model.addCons(y_vars[u] + y_vars[v] <= 1, name=f"edge_{u}_{v}")

        objective_expr = quicksum(x_vars[j] * c[j] + y_vars[j] * c[j] for j in range(self.n_cols))
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 1500,
        'n_cols': 843,
        'density': 0.1,
        'max_coef': 378,
        'n_nodes': 562,
        'm_parameter': 6,
    }

    complex_coverage_problem = ComplexCoverage(parameters, seed=seed)
    instance = complex_coverage_problem.generate_instance()
    solve_status, solve_time = complex_coverage_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")