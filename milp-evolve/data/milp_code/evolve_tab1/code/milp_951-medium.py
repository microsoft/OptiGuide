import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class ComplexSetCover:
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
        capacities = np.random.randint(self.min_capacity, self.max_capacity, size=self.n_cols)

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {'c': c, 
               'capacities': capacities,
               'indptr_csr': indptr_csr, 
               'indices_csr': indices_csr}

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        capacities = instance['capacities']

        model = Model("ComplexSetCover")
        var_names = {}

        # Create variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        y_vars = {f"y_{row}": model.addVar(vtype="I", lb=0, name=f"y_{row}") for row in range(self.n_rows)}

        # Add constraints to ensure a subset of rows are covered
        selected_rows = random.sample(range(self.n_rows), int(self.cover_fraction * self.n_rows))
        for row in selected_rows:
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        # Adding new type of constraint: Each row must be covered by a minimum number of unique columns
        for row in selected_rows:
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= self.min_cols_per_row, f"cover_{row}")

        # Adding capacity constraints for columns
        for j in range(self.n_cols):
            covered_rows = [row for row in selected_rows if j in indices_csr[indptr_csr[row]:indptr_csr[row + 1]]]
            model.addCons(quicksum(y_vars[f"y_{row}"] for row in covered_rows) <= capacities[j], f"capacity_{j}")

        # Set objective: Minimize total cost and balance load
        load_balance_expr = quicksum(y_vars[f"y_{row}"] for row in selected_rows)
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + load_balance_expr
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 3000,
        'n_cols': 500,
        'density': 0.73,
        'max_coef': 120,
        'min_cols_per_row': 10,
        'min_capacity': 25,
        'max_capacity': 250,
        'cover_fraction': 0.38,
    }

    complex_set_cover_problem = ComplexSetCover(parameters, seed=seed)
    instance = complex_set_cover_problem.generate_instance()
    solve_status, solve_time = complex_set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")