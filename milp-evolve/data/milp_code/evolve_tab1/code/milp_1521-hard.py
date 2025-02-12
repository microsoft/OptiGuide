import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class ArtExhibitAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_artworks * self.n_exhibits * self.showcase_density)

        # compute number of exhibits per artwork
        indices = np.random.choice(self.n_exhibits, size=nnzrs)  # random exhibit indexes
        indices[:2 * self.n_exhibits] = np.repeat(np.arange(self.n_exhibits), 2)  # force at least 2 exhibits per artwork
        _, exhibit_nartworks = np.unique(indices, return_counts=True)

        # for each exhibit, sample random artworks
        indices[:self.n_artworks] = np.random.permutation(self.n_artworks)  # force at least 1 exhibit per artwork
        i = 0
        indptr = [0]
        for n in exhibit_nartworks:
            if i >= self.n_artworks:
                indices[i:i + n] = np.random.choice(self.n_artworks, size=n, replace=False)
            elif i + n > self.n_artworks:
                remaining_artworks = np.setdiff1d(np.arange(self.n_artworks), indices[i:self.n_artworks], assume_unique=True)
                indices[self.n_artworks:i + n] = np.random.choice(remaining_artworks, size=i + n - self.n_artworks, replace=False)
            i += n
            indptr.append(i)

        # objective coefficients
        cost = np.random.randint(self.max_cost, size=self.n_exhibits) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_artworks, self.n_exhibits)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # Exhibit capacities, storage costs, and health constraints
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, size=self.n_exhibits)
        storage_cost = np.random.randint(self.min_storage_cost, self.max_storage_cost + 1, size=self.n_exhibits)
        health_req = np.random.randint(self.min_health, self.max_health + 1, size=self.n_exhibits)
        
        res = {'cost': cost,
               'indptr_csr': indptr_csr,
               'indices_csr': indices_csr,
               'capacities': capacities,
               'storage_cost': storage_cost,
               'health_req': health_req}

        # New instance data
        min_exhibit_time = np.random.randint(self.min_exhibit_time, self.max_exhibit_time + 1, size=self.n_exhibits)
        res['min_exhibit_time'] = min_exhibit_time
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        cost = instance['cost']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        capacities = instance['capacities']
        storage_cost = instance['storage_cost']
        health_req = instance['health_req']
        min_exhibit_time = instance['min_exhibit_time']

        model = Model("ArtExhibitAllocation")

        # Create variables for exhibit allocation and exhibiting time
        var_exhibit = {j: model.addVar(vtype="B", name=f"exhibit_{j}", obj=cost[j]) for j in range(self.n_exhibits)}
        var_exhibit_time = {j: model.addVar(vtype="C", name=f"exhibit_time_{j}", lb=0, ub=capacities[j]) for j in range(self.n_exhibits)}
        total_cost = model.addVar(vtype="I", name="total_cost")  # Total cost variable

        # Ensure each artwork's exhibit is met
        for artwork in range(self.n_artworks):
            exhibits = indices_csr[indptr_csr[artwork]:indptr_csr[artwork + 1]]
            model.addCons(quicksum(var_exhibit[j] for j in exhibits) >= 1, f"showcase_{artwork}")

        # Ensure exhibit capacity limit, storage cost, and health requirements
        for j in range(self.n_exhibits):
            model.addCons(var_exhibit_time[j] <= capacities[j], f"capacity_{j}")
            model.addCons(total_cost >= storage_cost[j], f"storage_cost_{j}")
            model.addCons(var_exhibit_time[j] >= var_exhibit[j] * min_exhibit_time[j], f"min_exhibit_time_{j}")  # Semi-continuous behavior
            model.addCons(var_exhibit_time[j] >= var_exhibit[j] * health_req[j], f"health_req_{j}")  # Health constraint

        # Objective: minimize total storing and exhibiting cost
        objective_expr = quicksum(var_exhibit[j] * cost[j] for j in range(self.n_exhibits)) + total_cost
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_artworks': 2000,
        'n_exhibits': 1400,
        'showcase_density': 0.1,
        'max_cost': 1500,
        'min_capacity': 700,
        'max_capacity': 1500,
        'min_storage_cost': 600,
        'max_storage_cost': 1800,
        'min_health': 175,
        'max_health': 500,
        'min_exhibit_time': 700,
        'max_exhibit_time': 700,
    }

    art_exhibit_allocation = ArtExhibitAllocation(parameters, seed=seed)
    instance = art_exhibit_allocation.generate_instance()
    solve_status, solve_time = art_exhibit_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")