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

        indices = np.random.choice(self.n_cols, size=nnzrs)  # random column indexes
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  # force at leats 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)

        indices[:self.n_rows] = np.random.permutation(self.n_rows) # force at least 1 column per row
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

        A = scipy.sparse.csc_matrix(
                (np.ones(len(indices), dtype=int), indices, indptr),
                shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        ### New Instance Data ###
        # Randomly generate column activation dependencies
        activation_matrix = np.zeros((self.n_cols, self.n_cols), dtype=int)
        for j in range(self.n_cols):
            if np.random.rand() > 0.5:  # 50% chance to create a dependency
                dependents = np.random.choice(np.delete(np.arange(self.n_cols), j), size=np.random.randint(1, 4), replace=False)
                activation_matrix[j, dependents] = 1

        res =  {'c': c, 
                'indptr_csr': indptr_csr, 
                'indices_csr': indices_csr,
                'activation_matrix': activation_matrix}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        activation_matrix = instance['activation_matrix']

        model = Model("SetCover")
        var_names = {}

        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        ### New Constraints ###
        big_M = 1e6
        for j in range(self.n_cols):
            dependents = np.where(activation_matrix[j] == 1)[0]
            for dep in dependents:
                model.addCons(var_names[j] >= var_names[dep] - big_M * (1 - var_names[j]), f"activation_{j}_{dep}")

        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
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
        'density': 0.17,
        'max_coef': 75,
    }
    ### New Parameter Code ###

    set_cover_problem = SetCover(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")