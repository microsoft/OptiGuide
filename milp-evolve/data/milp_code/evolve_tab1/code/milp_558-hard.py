import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class MedicalResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_neighborhoods * self.n_hospitals * self.density)

        # compute number of neighborhoods per hospital
        indices = np.random.choice(self.n_hospitals, size=nnzrs)
        indices[:2 * self.n_hospitals] = np.repeat(np.arange(self.n_hospitals), 2)

        _, hospital_neighborhoods = np.unique(indices, return_counts=True)
        indices[:self.n_neighborhoods] = np.random.permutation(self.n_neighborhoods)
        i = 0
        indptr = [0]
        for r in hospital_neighborhoods:
            if i >= self.n_neighborhoods:
                indices[i:i + r] = np.random.choice(self.n_neighborhoods, size=r, replace=False)
            elif i + r > self.n_neighborhoods:
                remaining_neighborhoods = np.setdiff1d(np.arange(self.n_neighborhoods), indices[i:self.n_neighborhoods], assume_unique=True)
                indices[self.n_neighborhoods:i + r] = np.random.choice(remaining_neighborhoods, size=i + r - self.n_neighborhoods, replace=False)
            i += r
            indptr.append(i)

        # doctors availability and medicine costs
        doctors = np.random.randint(self.max_doctors, size=self.n_hospitals) + 1
        medcosts = np.random.randint(self.max_medcost, size=self.n_hospitals) + 1

        carelevels = np.random.randint(self.max_carelevel, size=self.n_neighborhoods) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix((np.ones(len(indices), dtype=int), indices, indptr), shape=(self.n_neighborhoods, self.n_hospitals)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {
            'doctors': doctors,
            'medcosts': medcosts,
            'carelevels': carelevels,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        doctors = instance['doctors']
        medcosts = instance['medcosts']
        carelevels = instance['carelevels']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']

        model = Model("MedicalResourceAllocation")
        var_names = {}

        # Create variables and set objective
        for j in range(self.n_hospitals):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=medcosts[j])

        # Add constraints to ensure care levels are met and doctors are not overburdened
        for neighborhood in range(self.n_neighborhoods):
            hospitals = indices_csr[indptr_csr[neighborhood]:indptr_csr[neighborhood + 1]]
            model.addCons(
                quicksum(var_names[j] * doctors[j] for j in hospitals) >= carelevels[neighborhood], f"carelevel_{neighborhood}"
            )

        # Set objective: Minimize total medicine cost
        objective_expr = quicksum(var_names[j] * medcosts[j] for j in range(self.n_hospitals))
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_neighborhoods': 1500,
        'n_hospitals': 750,
        'density': 0.38,
        'max_doctors': 250,
        'max_medcost': 200,
        'max_carelevel': 5,
    }

    medical_allocation_problem = MedicalResourceAllocation(parameters, seed=seed)
    instance = medical_allocation_problem.generate_instance()
    solve_status, solve_time = medical_allocation_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")