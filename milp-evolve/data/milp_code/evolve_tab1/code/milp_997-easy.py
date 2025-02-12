import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class SetCoverWithSupplyChain:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
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

        c = np.random.randint(self.max_coef // 4, size=self.n_cols) + 1

        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr
        
        facilities = np.random.choice(range(1000), size=self.n_facilities, replace=False)
        demand_nodes = np.random.choice(range(1000, 2000), size=self.n_demand_nodes, replace=False)
        demands = {node: np.random.randint(self.min_demand, self.max_demand) for node in demand_nodes}
        transport_costs = {(f, d): np.random.uniform(self.min_transport_cost, self.max_transport_cost) for f in facilities for d in demand_nodes}
        capacities = {f: np.random.randint(self.min_capacity, self.max_capacity) for f in facilities}
        facility_costs = {f: np.random.uniform(self.min_facility_cost, self.max_facility_cost) for f in facilities}
        processing_costs = {f: np.random.uniform(self.min_processing_cost, self.max_processing_cost) for f in facilities}
        auxiliary_costs = {f: np.random.uniform(self.min_auxiliary_cost, self.max_auxiliary_cost) for f in facilities}

        res = {
            'c': c, 
            'indices_csr': indices_csr, 
            'indptr_csr': indptr_csr,
            'facilities': facilities,
            'demand_nodes': demand_nodes,
            'transport_costs': transport_costs,
            'capacities': capacities,
            'demands': demands,
            'facility_costs': facility_costs,
            'processing_costs': processing_costs,
            'auxiliary_costs': auxiliary_costs,
        }
        return res
    
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        facilities = instance['facilities']
        demand_nodes = instance['demand_nodes']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        facility_costs = instance['facility_costs']
        processing_costs = instance['processing_costs']
        auxiliary_costs = instance['auxiliary_costs']

        model = Model("SetCoverWithSupplyChain")
        var_names = {}
        Facility_location_vars = {f: model.addVar(vtype="B", name=f"F_Loc_{f}") for f in facilities}
        Allocation_vars = {(f, d): model.addVar(vtype="B", name=f"Alloc_{f}_{d}") for f in facilities for d in demand_nodes}
        Transportation_vars = {(f, d): model.addVar(vtype="C", name=f"Trans_{f}_{d}") for f in facilities for d in demand_nodes}
        Assembly_vars = {f: model.addVar(vtype="C", name=f"Assembly_{f}") for f in facilities}
        Ancillary_vars = {f: model.addVar(vtype="C", name=f"Ancillary_{f}") for f in facilities}
        
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        selected_rows = random.sample(range(self.n_rows), int(self.cover_fraction * self.n_rows))
        for row in selected_rows:
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        
        total_cost = quicksum(Facility_location_vars[f] * facility_costs[f] for f in facilities)
        total_cost += quicksum(Allocation_vars[f, d] * transport_costs[f, d] for f in facilities for d in demand_nodes)
        total_cost += quicksum(Assembly_vars[f] * processing_costs[f] for f in facilities)
        total_cost += quicksum(Ancillary_vars[f] * auxiliary_costs[f] for f in facilities)
        
        for f in facilities:
            model.addCons(
                quicksum(Transportation_vars[f, d] for d in demand_nodes) <= capacities[f] * Facility_location_vars[f],
                name=f"Capacity_{f}"
            )

        for d in demand_nodes:
            model.addCons(
                quicksum(Transportation_vars[f, d] for f in facilities) >= demands[d],
                name=f"DemandSatisfaction_{d}"
            )

        for f in facilities:
            model.addCons(
                Assembly_vars[f] <= capacities[f],
                name=f"AssemblyLimit_{f}"
            )

        for f in facilities:
            model.addCons(
                Ancillary_vars[f] <= capacities[f] * 0.1,  # Assume ancillary operations are limited to 10% of the capacity
                name=f"AncillaryLimit_{f}"
            )

        model.setObjective(objective_expr + total_cost, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 3000,
        'n_cols': 750,
        'density': 0.45,
        'max_coef': 150,
        'cover_fraction': 0.66,
        'n_facilities': 100,
        'n_demand_nodes': 5,
        'min_demand': 20,
        'max_demand': 525,
        'min_transport_cost': 5.0,
        'max_transport_cost': 500.0,
        'min_capacity': 100,
        'max_capacity': 2500,
        'min_facility_cost': 50000,
        'max_facility_cost': 200000,
        'min_processing_cost': 1000,
        'max_processing_cost': 1000,
        'min_auxiliary_cost': 200,
        'max_auxiliary_cost': 600,
    }

    set_cover_problem = SetCoverWithSupplyChain(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")