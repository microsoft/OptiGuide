import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class SetCoverFacilityLocationService:
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
        
        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost, (self.n_facilities, self.n_cols))
        capacities = np.random.randint(self.min_capacity, self.max_capacity, self.n_facilities)

        service_levels = np.random.randint(self.min_service_level, self.max_service_level, self.n_facilities)
        service_times = np.random.randint(self.min_service_time, self.max_service_time, self.n_facilities)

        res = {
            'c': c,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'crucial_sets': crucial_sets,
            'activation_cost': activation_cost,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs,
            'capacities': capacities,
            'service_levels': service_levels,
            'service_times': service_times
        }
        return res

    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        crucial_sets = instance['crucial_sets']
        activation_cost = instance['activation_cost']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        service_levels = instance['service_levels']
        service_times = instance['service_times']

        model = Model("SetCoverFacilityLocationService")
        var_names = {}
        activate_crucial = {}
        facility_vars = {}
        allocation_vars = {}
        service_vars = {}

        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        for idx, j in enumerate(crucial_sets):
            activate_crucial[j] = model.addVar(vtype="B", name=f"y_{j}", obj=activation_cost[idx])

        for f in range(self.n_facilities):
            facility_vars[f] = model.addVar(vtype="B", name=f"Facility_{f}", obj=fixed_costs[f])
            service_vars[f] = model.addVar(vtype="C", name=f"Service_{f}", lb=0, ub=service_levels[f], obj=service_times[f])
            for j in range(self.n_cols):
                allocation_vars[(f, j)] = model.addVar(vtype="B", name=f"Facility_{f}_Column_{j}")

        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"Cover_Row_{row}")

        for j in crucial_sets:
            rows_impacting_j = np.where(indices_csr == j)[0]
            for row in rows_impacting_j:
                model.addCons(var_names[j] >= activate_crucial[j], f"Crucial_Coverage_Row_{row}_Set_{j}")

        for f in range(self.n_facilities):
            model.addCons(quicksum(allocation_vars[(f, j)] for j in range(self.n_cols)) <= capacities[f], f"Facility_{f}_Capacity")
            model.addCons(quicksum(allocation_vars[(f, j)] * transport_costs[f][j] for j in range(self.n_cols)) <= service_vars[f], f"Facility_{f}_Service")
            for j in range(self.n_cols):
                model.addCons(allocation_vars[(f, j)] <= facility_vars[f], f"Facility_{f}_Alloc_{j}")

        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + \
                         quicksum(activate_crucial[j] * activation_cost[idx] for idx, j in enumerate(crucial_sets)) + \
                         quicksum(fixed_costs[f] * facility_vars[f] for f in range(self.n_facilities)) + \
                         quicksum(transport_costs[f][j] * allocation_vars[(f, j)] for f in range(self.n_facilities) for j in range(self.n_cols)) + \
                         quicksum(service_times[f] * service_vars[f] for f in range(self.n_facilities))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 1125,
        'n_cols': 2250,
        'density': 0.73,
        'max_coef': 7,
        'n_crucial': 22,
        'activation_cost_low': 13,
        'activation_cost_high': 1000,
        'n_facilities': 45,
        'min_fixed_cost': 270,
        'max_fixed_cost': 592,
        'min_transport_cost': 1620,
        'max_transport_cost': 2401,
        'min_capacity': 714,
        'max_capacity': 1138,
        'min_service_level': 50,
        'max_service_level': 200,
        'min_service_time': 1,
        'max_service_time': 10
    }

    problem = SetCoverFacilityLocationService(parameters, seed=seed)
    instance = problem.generate_instance()
    solve_status, solve_time = problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")