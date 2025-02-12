import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class LibraryResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_users * self.n_resources * self.demand_density)

        # compute number of users per resource
        indices = np.random.choice(self.n_resources, size=nnzrs)  # random resource indexes
        indices[:2 * self.n_resources] = np.repeat(np.arange(self.n_resources), 2)  # force at least 2 users per resource
        _, res_nusers = np.unique(indices, return_counts=True)

        # for each resource, sample random users
        indices[:self.n_users] = np.random.permutation(self.n_users)  # force at least 1 resource per user
        i = 0
        indptr = [0]
        for n in res_nusers:
            if i >= self.n_users:
                indices[i:i + n] = np.random.choice(self.n_users, size=n, replace=False)
            elif i + n > self.n_users:
                remaining_users = np.setdiff1d(np.arange(self.n_users), indices[i:self.n_users], assume_unique=True)
                indices[self.n_users:i + n] = np.random.choice(remaining_users, size=i + n - self.n_users, replace=False)
            i += n
            indptr.append(i)

        # objective coefficients
        c = np.random.randint(self.max_cost, size=self.n_resources) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_users, self.n_resources)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # Resource capacities
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, size=self.n_resources)
        
        res = {'c': c,
               'indptr_csr': indptr_csr,
               'indices_csr': indices_csr,
               'capacities': capacities}

        # New instance data
        min_usage = np.random.randint(self.min_resource_usage, self.max_resource_usage + 1, size=self.n_resources)
        res['min_usage'] = min_usage
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        capacities = instance['capacities']
        min_usage = instance['min_usage']

        model = Model("LibraryResourceAllocation")

        # Create variables for resource allocation and usage
        var_allocation = {j: model.addVar(vtype="B", name=f"allocation_{j}", obj=c[j]) for j in range(self.n_resources)}
        var_usage = {j: model.addVar(vtype="C", name=f"usage_{j}", lb=0, ub=capacities[j]) for j in range(self.n_resources)}

        # Ensure each user's demand is met
        for user in range(self.n_users):
            resources = indices_csr[indptr_csr[user]:indptr_csr[user + 1]]
            model.addCons(quicksum(var_allocation[j] for j in resources) >= 1, f"demand_{user}")

        # Ensure resource capacity limit
        for j in range(self.n_resources):
            model.addCons(var_usage[j] <= capacities[j], f"capacity_{j}")
            model.addCons(var_usage[j] >= var_allocation[j] * min_usage[j], f"min_usage_{j}")  # Semi-continuous behavior

        # Objective: minimize total cost
        objective_expr = quicksum(var_allocation[j] * c[j] for j in range(self.n_resources))
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_users': 1500,
        'n_resources': 2800,
        'demand_density': 0.1,
        'max_cost': 1500,
        'min_capacity': 700,
        'max_capacity': 1500,
        'min_resource_usage': 350,
        'max_resource_usage': 2800,
    }

    library_resource_allocation = LibraryResourceAllocation(parameters, seed=seed)
    instance = library_resource_allocation.generate_instance()
    solve_status, solve_time = library_resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")