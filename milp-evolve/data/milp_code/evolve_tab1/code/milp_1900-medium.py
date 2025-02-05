import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DiverseSetCover:
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

        crucial_sets = np.random.choice(self.n_cols, self.n_crucial, replace=False)
        activation_cost = np.random.randint(self.activation_cost_low, self.activation_cost_high, size=self.n_crucial)

        resource_cap = np.random.randint(self.resource_cap_low, self.resource_cap_high, size=(self.n_cols, self.n_resources))
        resource_limit = np.random.randint(1, np.sum(resource_cap, axis=0) / 2, size=self.n_resources)

        penalties = np.random.randint(self.penalty_low, self.penalty_high, size=self.n_rows)

        res = {
            'c': c,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'crucial_sets': crucial_sets,
            'activation_cost': activation_cost,
            'resource_cap': resource_cap,
            'resource_limit': resource_limit,
            'penalties': penalties
        }
        return res

    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        crucial_sets = instance['crucial_sets']
        activation_cost = instance['activation_cost']
        resource_cap = instance['resource_cap']
        resource_limit = instance['resource_limit']
        penalties = instance['penalties']

        model = Model("DiverseSetCover")
        var_names = {}
        activate_crucial = {}
        resource_usage = {}

        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        for idx, j in enumerate(crucial_sets):
            activate_crucial[j] = model.addVar(vtype="B", name=f"y_{j}", obj=activation_cost[idx])

        for r in range(self.n_resources):
            resource_usage[r] = model.addVar(vtype="C", name=f"resource_{r}")

        not_covered_penalty = {}
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            not_covered_penalty[row] = model.addVar(vtype="B", name=f"penalty_{row}", obj=penalties[row])
            model.addCons(quicksum(var_names[j] for j in cols) >= 1 - not_covered_penalty[row], f"c_{row}")

        for j in crucial_sets:
            rows_impacting_j = np.where(indices_csr == j)[0]
            for row in rows_impacting_j:
                model.addCons(var_names[j] >= activate_crucial[j], f"crucial_coverage_row_{row}_set_{j}")

        for r in range(self.n_resources):
            model.addCons(quicksum(var_names[j] * resource_cap[j, r] for j in range(self.n_cols)) <= resource_limit[r], f"resource_limit_{r}")

        budget_constraint = model.addCons(quicksum(var_names[j] for j in range(self.n_cols)) <= self.budget, "budget_constraint")

        # Adding an example knapsack constraint
        example_item_weights = np.random.randint(1, 10, size=self.n_cols)
        knapsack_limit = np.random.randint(10, 20)
        model.addCons(quicksum(var_names[j] * example_item_weights[j] for j in range(self.n_cols)) <= knapsack_limit, "knapsack_constraint")

        for r in resource_usage:
            model.addCons(resource_usage[r] >= 0, f"non_negative_resource_usage_{r}")

        # Dual-Objective: Minimize cost and penalty with resource constraints
        objective_expr = (
            quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + 
            quicksum(activate_crucial[j] * activation_cost[idx] for idx, j in enumerate(crucial_sets)) +
            quicksum(not_covered_penalty[row] * penalties[row] for row in range(self.n_rows))
        )
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 562,
        'n_cols': 750,
        'density': 0.1,
        'max_coef': 21,
        'n_crucial': 630,
        'activation_cost_low': 315,
        'activation_cost_high': 1125,
        'n_resources': 90,
        'resource_cap_low': 0,
        'resource_cap_high': 30,
        'penalty_low': 250,
        'penalty_high': 600,
        'budget': 1500,
    }

    diverse_set_cover_problem = DiverseSetCover(parameters, seed=seed)
    instance = diverse_set_cover_problem.generate_instance()
    solve_status, solve_time = diverse_set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")