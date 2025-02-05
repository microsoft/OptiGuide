import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class SimpleEnergyDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_regions * self.n_plants * self.density)

        # compute number of regions per plant
        indices = np.random.choice(self.n_plants, size=nnzrs)
        indices[:2 * self.n_plants] = np.repeat(np.arange(self.n_plants), 2)

        _, plant_regions = np.unique(indices, return_counts=True)
        indices[:self.n_regions] = np.random.permutation(self.n_regions)
        i = 0
        indptr = [0]
        for r in plant_regions:
            if i >= self.n_regions:
                indices[i:i+r] = np.random.choice(self.n_regions, size=r, replace=False)
            elif i + r > self.n_regions:
                remaining_regions = np.setdiff1d(np.arange(self.n_regions), indices[i:self.n_regions], assume_unique=True)
                indices[self.n_regions:i+r] = np.random.choice(remaining_regions, size=i+r-self.n_regions, replace=False)
            i += r
            indptr.append(i)

        # power production capabilities and costs
        capacities = np.random.randint(self.max_capacity, size=self.n_plants) + 1
        costs = np.random.randint(self.max_cost, size=self.n_plants) + 1

        demands = np.random.randint(self.max_demand, size=self.n_regions) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
                (np.ones(len(indices), dtype=int), indices, indptr),
                shape=(self.n_regions, self.n_plants)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res =  {'capacities': capacities,
                'costs': costs, 
                'demands': demands,
                'indptr_csr': indptr_csr, 
                'indices_csr': indices_csr}

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        capacities = instance['capacities']
        costs = instance['costs']
        demands = instance['demands']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']

        model = Model("SimpleEnergyDistribution")
        var_names = {}

        # Create variables and set objective
        for j in range(self.n_plants):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=costs[j])

        # Add constraints to ensure demand is met
        for region in range(self.n_regions):
            plants = indices_csr[indptr_csr[region]:indptr_csr[region + 1]]
            model.addCons(
                quicksum(var_names[j] * capacities[j] for j in plants) >= demands[region], f"demand_{region}"
            )

        # Set objective: Minimize total cost
        objective_expr = quicksum(var_names[j] * costs[j] for j in range(self.n_plants))
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_regions': 600,
        'n_plants': 800,
        'density': 0.6,
        'max_capacity': 1000,
        'max_cost': 900,
        'max_demand': 750,
    }

    energy_problem = SimpleEnergyDistribution(parameters, seed=seed)
    instance = energy_problem.generate_instance()
    solve_status, solve_time = energy_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")