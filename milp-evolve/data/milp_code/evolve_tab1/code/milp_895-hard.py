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

        # Additional manufacturing constraints data
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        nominal_capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        uncertainty_capacity = np.random.randint(5, 20, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            col1 = random.randint(0, self.n_cols - 1)
            col2 = random.randint(0, self.n_cols - 1)
            if col1 != col2:
                mutual_exclusivity_pairs.append((col1, col2))

        res.update({
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "nominal_capacity": nominal_capacity,
            "uncertainty_capacity": uncertainty_capacity,
            "setup_cost": setup_cost,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs})

        ### given instance data code ends here
        ### new instance data code ends here
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        nominal_capacity = instance['nominal_capacity']
        uncertainty_capacity = instance['uncertainty_capacity']
        setup_cost = instance['setup_cost']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']

        model = Model("SetCover")
        var_names = {}
        open_facility = {j: model.addVar(vtype="B", name=f"OpenFacility_{j}") for j in range(n_facilities)}
        col_to_facility = {(col, f): model.addVar(vtype="B", name=f"ColFacility_{col}_{f}") for col in range(self.n_cols) for f in range(n_facilities)}

        # Create variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        # Add constraints to ensure a subset of rows are covered
        selected_rows = random.sample(range(self.n_rows), int(self.cover_fraction * self.n_rows))
        for row in selected_rows:
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        # Robust facility capacity constraints
        for f in range(n_facilities):
            rhs = nominal_capacity[f] - uncertainty_capacity[f]
            model.addCons(quicksum(col_to_facility[col, f] for col in range(self.n_cols)) <= rhs + uncertainty_capacity[f] * (1 - open_facility[f]), f"RobustFacilityCapacity_{f}")

        # Column assignment to facility
        for col in range(self.n_cols):
            model.addCons(quicksum(col_to_facility[col, f] for f in range(n_facilities)) == var_names[col], f"ColFacility_{col}")

        # Mutual exclusivity constraints
        for (col1, col2) in mutual_exclusivity_pairs:
            model.addCons(var_names[col1] + var_names[col2] <= 1, f"Exclusive_{col1}_{col2}")

        # Set objective: Minimize total cost
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) \
                         + quicksum(operating_cost[f] * open_facility[f] for f in range(n_facilities)) \
                         + quicksum(setup_cost[f] * open_facility[f] for f in range(n_facilities))
        
        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
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
        'density': 0.17,
        'max_coef': 1200,
        'cover_fraction': 0.52,
        'facility_min_count': 3,
        'facility_max_count': 15,
        'n_exclusive_pairs': 2,
        'uncertainty': 0.1
    }
    ### given parameter code ends here
    ### new parameter code ends here

    set_cover_problem = SetCover(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")