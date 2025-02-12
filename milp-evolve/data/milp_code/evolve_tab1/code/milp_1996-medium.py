import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class RobustResourceAllocation:
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
        indices[:self.n_rows] = np.random.permutation(self.n_rows)  # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= self.n_rows:
                indices[i:i + n] = np.random.choice(self.n_rows, size=n, replace=False)
            # partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i + n] = np.random.choice(remaining_rows, size=i + n - self.n_rows, replace=False)
            i += n
            indptr.append(i)

        # objective coefficients for set cover
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # Generate uncertain demands as average demand ± deviation
        avg_demand = np.random.randint(self.demand_low, self.demand_high, size=self.n_cols)
        demand_deviation = np.random.randint(1, self.max_deviation, size=self.n_cols)

        res = {
            'c': c,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'avg_demand': avg_demand,
            'demand_deviation': demand_deviation,
            'capacity': self.total_capacity,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        avg_demand = instance['avg_demand']
        demand_deviation = instance['demand_deviation']
        capacity = instance['capacity']

        model = Model("RobustResourceAllocation")
        var_names = {}
        demand_buffers = {}

        # Create variables and set objective for resource allocation
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])
            demand_buffers[j] = model.addVar(vtype="C", name=f"demand_buffer_{j}")

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"Cover_Row_{row}")

        # Add robust constraints ensuring that selection can handle the worst-case demand within specified deviation
        for j in range(self.n_cols):
            model.addCons(var_names[j] * (avg_demand[j] + demand_buffers[j]) <= avg_demand[j] + demand_deviation[j], f"Robust_Demand_{j}")

        # Ensure total capacity constraint accounting for buffers
        model.addCons(quicksum(var_names[j] * avg_demand[j] + demand_buffers[j] for j in range(self.n_cols)) <= capacity, "Total_Capacity_Constraint")

        # Objective: Minimize total cost including set cover costs and buffers
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + \
                         quicksum(demand_buffers[j] for j in range(self.n_cols))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 112,
        'n_cols': 843,
        'density': 0.24,
        'max_coef': 3,
        'demand_low': 0,
        'demand_high': 70,
        'total_capacity': 5000,
        'max_deviation': 200,
    }
    problem = RobustResourceAllocation(parameters, seed=seed)
    instance = problem.generate_instance()
    solve_status, solve_time = problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")