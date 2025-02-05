import random
import time
import scipy
import numpy as np
import networkx as nx
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

        # Compute number of rows per column
        indices = np.random.choice(self.n_cols, size=nnzrs)  # random column indexes
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  # force at least 2 rows per column
        _, col_nrows = np.unique(indices, return_counts=True)

        # For each column, sample random rows
        indices[:self.n_rows] = np.random.permutation(self.n_rows)  # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # Empty column, fill with random rows
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)

            # Partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)

            i += n
            indptr.append(i)

        # Objective coefficients
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # Sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
                (np.ones(len(indices), dtype=int), indices, indptr),
                shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {'c': c, 
               'indptr_csr': indptr_csr, 
               'indices_csr': indices_csr}

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']

        model = Model("SetCover")
        var_names = {}
        uncovered_penalty = 1000  # Penalty for uncovered rows

        # Create variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        uncovered_rows = []
        for row in range(self.n_rows):
            # Randomly decide whether to enforce the coverage constraint
            if random.random() < self.coverage_probability:
                cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
                model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")
            else:
                uncovered_rows.append(row)

        # Model uncovered rows penalty
        uncovered_vars = {}
        for row in uncovered_rows:
            uncovered_vars[row] = model.addVar(vtype="B", name=f"uncovered_{row}")
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1 - uncovered_vars[row])

        # Objective: Minimize total cost + penalty for uncovered rows
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + \
                         quicksum(uncovered_vars[row] * uncovered_penalty for row in uncovered_rows)

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 2250,
        'n_cols': 1500,
        'density': 0.1,
        'max_coef': 700,
        'coverage_probability': 0.73,
    }

    set_cover_problem = SetCover(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")