import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FleetManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_locs * self.n_vehicles * self.density)

        # compute number of rows per columns
        indices = np.random.choice(self.n_vehicles, size=nnzrs)
        indices[:2 * self.n_vehicles] = np.repeat(np.arange(self.n_vehicles), 2)
        _, col_nrows = np.unique(indices, return_counts=True)

        indices[:self.n_locs] = np.random.permutation(self.n_locs)
        i = 0
        indptr = [0]
        for n in col_nrows:
            if i >= self.n_locs:
                indices[i:i+n] = np.random.choice(self.n_locs, size=n, replace=False)
            elif i + n > self.n_locs:
                remaining_rows = np.setdiff1d(np.arange(self.n_locs), indices[i:self.n_locs], assume_unique=True)
                indices[self.n_locs:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_locs, replace=False)
            i += n
            indptr.append(i)

        vehicle_costs = np.random.randint(self.max_vehicle_cost, size=self.n_vehicles) + 1
        maintenance_costs = np.random.randint(self.max_maintenance_cost, size=self.n_vehicles) + 1
        delay_penalties = np.random.randint(self.max_delay_penalty, size=self.n_routes) + 1
        A = scipy.sparse.csc_matrix((np.ones(len(indices), dtype=int), indices, indptr), shape=(self.n_locs, self.n_vehicles)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {
            'vehicle_costs': vehicle_costs, 
            'maintenance_costs': maintenance_costs,
            'delay_penalties': delay_penalties,
            'indptr_csr': indptr_csr, 
            'indices_csr': indices_csr
        }
        ### new instance data code ends here
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        vehicle_costs = instance['vehicle_costs']
        maintenance_costs = instance['maintenance_costs']
        delay_penalties = instance['delay_penalties']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']

        model = Model("FleetManagement")
        vehicle_vars = {}
        maintenance_vars = {}
        delay_vars = {}

        # Create variables and set objective
        for j in range(self.n_vehicles):
            vehicle_vars[j] = model.addVar(vtype="B", name=f"vehicle_{j}", obj=vehicle_costs[j])

        for v in range(self.n_vehicles):
            maintenance_vars[v] = model.addVar(vtype="C", name=f"maintenance_{v}", obj=maintenance_costs[v])

        for r in range(self.n_routes):
            delay_vars[r] = model.addVar(vtype="C", name=f"delay_{r}", obj=delay_penalties[r])

        # Add constraints to ensure each location is covered
        for loc in range(self.n_locs):
            vehicles = indices_csr[indptr_csr[loc]:indptr_csr[loc + 1]]
            model.addCons(quicksum(vehicle_vars[j] for j in vehicles) >= 1, f"coverage_constraint_{loc}")
            model.addCons(quicksum(vehicle_vars[j] for j in vehicles) + maintenance_vars[loc % self.n_vehicles] >= 1, f"maintenance_schedule_{loc}")

        # Set objective: Minimize total cost plus delays penalties
        objective_expr = quicksum(vehicle_vars[j] * vehicle_costs[j] for j in range(self.n_vehicles)) + \
                         quicksum(maintenance_vars[v] * maintenance_costs[v] for v in range(self.n_vehicles)) + \
                         quicksum(delay_vars[r] * delay_penalties[r] for r in range(self.n_routes))
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locs': 1500,
        'n_vehicles': 562,
        'density': 0.17,
        'max_vehicle_cost': 37,
        'max_maintenance_cost': 37,
        'max_delay_penalty': 450,
        'n_routes': 2000,
    }
    ### new parameter code ends here

    fleet_management = FleetManagement(parameters, seed=seed)
    instance = fleet_management.generate_instance()
    solve_status, solve_time = fleet_management.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")