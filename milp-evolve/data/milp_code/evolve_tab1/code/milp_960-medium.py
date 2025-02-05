import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class SetCover:
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
        indices = np.random.choice(self.n_cols, size=nnzrs, replace=True)  # random column indexes with replacement
        _, col_nrows = np.unique(indices, return_counts=True)

        # Randomly assign rows and columns
        indices[:self.n_rows] = np.random.permutation(self.n_rows)  # ensure some rows are covered
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

        # objective coefficients
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {'c': c, 
               'indptr_csr': indptr_csr, 
               'indices_csr': indices_csr}

        # Additional logical condition data
        self.logic_conditions = np.random.randint(2, size=(self.n_logic_rows, self.n_cols)).tolist()

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']

        model = Model("SetCover")
        var_names = {}
        logic_vars = {}

        # Create variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        # Add constraints to ensure a subset of rows are covered
        selected_rows = random.sample(range(self.n_rows), int(self.cover_fraction * self.n_rows))
        for row in selected_rows:
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        # Set objective: Minimize total cost
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        
        # Introducing logical conditions
        for i in range(self.n_logic_rows):
            for j in range(self.n_cols):
                logic_vars[i, j] = model.addVar(vtype="B", name=f"y_{i}_{j}")
                # Adding logical constraints
                model.addCons(var_names[j] - logic_vars[i, j] >= 0)  # x_j must be 1 for y_ij to be 1
                model.addCons(logic_vars[i, j] <= self.logic_conditions[i][j])  # y_ij only if condition met

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 3000,
        'n_cols': 375,
        'density': 0.1,
        'max_coef': 900,
        'cover_fraction': 0.52,
        'n_logic_rows': 50,
    }
    set_cover_problem = SetCover(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")