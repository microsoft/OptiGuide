import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedSetCover:
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

        # compute number of rows per column
        indices = np.random.choice(self.n_cols, size=nnzrs)
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)
        _, col_nrows = np.unique(indices, return_counts=True)

        indices[:self.n_rows] = np.random.permutation(self.n_rows)
        i = 0
        indptr = [0]
        for n in col_nrows:
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)
            i += n
            indptr.append(i)

        c = np.random.randint(self.max_coef, size=self.n_cols) + 1
        penalties = np.random.randint(self.max_penalty, size=self.n_rows) + 1
        A = scipy.sparse.csc_matrix((np.ones(len(indices), dtype=int), indices, indptr), shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {
            'c': c, 
            'penalties': penalties,
            'indptr_csr': indptr_csr, 
            'indices_csr': indices_csr
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        penalties = instance['penalties']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']

        model = Model("EnhancedSetCover")
        var_names = {}
        penalties_vars = {}

        # Create variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        for row in range(self.n_rows):
            penalties_vars[row] = model.addVar(vtype="C", name=f"penalty_{row}", obj=penalties[row])

        # Add constraints to ensure each row is covered within bounds
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"min_c_{row}")
            model.addCons(quicksum(var_names[j] for j in cols) + penalties_vars[row] >= 1, f"penalty_c_{row}")

        # Set objective: Minimize total cost plus penalties
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + quicksum(penalties_vars[row] * penalties[row] for row in range(self.n_rows))
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 1500,
        'n_cols': 1125,
        'density': 0.24,
        'max_coef': 50,
        'max_penalty': 25,
    }

    enhanced_set_cover = EnhancedSetCover(parameters, seed=seed)
    instance = enhanced_set_cover.generate_instance()
    solve_status, solve_time = enhanced_set_cover.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")