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

        # compute number of rows per column
        indices = np.random.choice(self.n_cols, size=nnzrs)  # random column indexes
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  # force at least 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)

        # for each column, sample random rows
        indices[:self.n_rows] = np.random.permutation(self.n_rows) # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)

            # partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)

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

        res =  {'c': c, 
                'indptr_csr': indptr_csr, 
                'indices_csr': indices_csr}

        # New instance data generation: partial budgets and investment priorities
        partial_budgets = np.random.uniform(1, 5, size=self.n_cols)
        investment_priority = np.random.uniform(1, 100, size=self.n_cols)
        res.update({'partial_budgets': partial_budgets, 'investment_priority': investment_priority})

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']

        partial_budgets = instance['partial_budgets']
        investment_priority = instance['investment_priority']

        model = Model("SetCover")
        var_names = {}
        investment_levels = {}

        # Create variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])
            investment_levels[j] = [
                model.addVar(vtype="C", lb=0, ub=self.investment_max, name=f"investment_{j}_{i}") for i in range(3)
            ]

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        # Adding SOS2 constraints for investment levels
        for j in range(self.n_cols):
            model.addConsSOS2([investment_levels[j][i] for i in range(3)], name=f"sos2_{j}")

        # Objective: Minimize total cost plus investment priorities
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + \
                         quicksum(investment_levels[j][0] * investment_priority[j] for j in range(self.n_cols))

        # Budget constraint
        total_investment_cost = quicksum(var_names[j] * sum(investment_levels[j]) * partial_budgets[j]
                                         for j in range(self.n_cols))
        model.addCons(total_investment_cost <= self.budget_limit, name="budget_limit")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 750,
        'n_cols': 1500,
        'density': 0.05,
        'max_coef': 100,
        'investment_max': 50,  # New parameter
        'budget_limit': 10000   # New parameter
    }

    set_cover_problem = SetCover(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")