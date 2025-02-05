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
        
        # Generate new data for enhanced complexity
        seasons = ["Spring", "Summer", "Fall", "Winter"]
        season_demands = {season: {node: int(demands[node] * np.random.uniform(0.5, 1.5)) for node in demand_nodes} for season in seasons}
        season_holding_costs = {season: {f: np.random.uniform(self.min_holding_cost, self.max_holding_cost) for f in facilities} for season in seasons}
        lead_time_variability = {mode: np.random.uniform(self.min_lead_time, self.max_lead_time) for mode in self.transportation_modes}
        carbon_footprints = {mode: np.random.uniform(self.min_carbon, self.max_carbon) for mode in self.transportation_modes}
        mode_capacities = {mode: np.random.randint(self.min_mode_capacity, self.max_mode_capacity) for mode in self.transportation_modes}

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
            'season_demands': season_demands,
            'season_holding_costs': season_holding_costs,
            'lead_time_variability': lead_time_variability,
            'carbon_footprints': carbon_footprints,
            'mode_capacities': mode_capacities,
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
        season_demands = instance['season_demands']
        season_holding_costs = instance['season_holding_costs']
        lead_time_variability = instance['lead_time_variability']
        carbon_footprints = instance['carbon_footprints']
        mode_capacities = instance['mode_capacities']
        
        model = Model("SetCoverWithSupplyChain")
        var_names = {}
        Facility_location_vars = {f: model.addVar(vtype="B", name=f"F_Loc_{f}") for f in facilities}
        Allocation_vars = {(f, d): model.addVar(vtype="B", name=f"Alloc_{f}_{d}") for f in facilities for d in demand_nodes}
        Transportation_vars = {(f, d): model.addVar(vtype="C", name=f"Trans_{f}_{d}") for f in facilities for d in demand_nodes}
        Assembly_vars = {f: model.addVar(vtype="C", name=f"Assembly_{f}") for f in facilities}
        Ancillary_vars = {f: model.addVar(vtype="C", name=f"Ancillary_{f}") for f in facilities}
        Mode_vars = {(m, f, d): model.addVar(vtype="C", name=f"Mode_{m}_{f}_{d}") for m in self.transportation_modes for f in facilities for d in demand_nodes}
        
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
        
        # New constraints
        for season in season_demands:
            for d in demand_nodes:
                model.addCons(
                    quicksum(Transportation_vars[f, d] for f in facilities) >= season_demands[season][d],
                    name=f"SeasonalDemand_{season}_{d}"
                )
        for season in season_holding_costs:
            for f in facilities:
                model.addCons(
                    Ancillary_vars[f] <= capacities[f] * Facility_location_vars[f] * 0.15,  # Assuming ancillary operations for seasonal limit
                    name=f"SeasonalAncillary_{season}_{f}"
                )
        
        for m in self.transportation_modes:
            for f in facilities:
                for d in demand_nodes:
                    model.addCons(
                        Mode_vars[m, f, d] <= lead_time_variability[m],
                        name=f"LeadTime_{m}_{f}_{d}"
                    )
        
        carbon_cost = quicksum(Mode_vars[m, f, d] * carbon_footprints[m] for m in self.transportation_modes for f in facilities for d in demand_nodes)
        total_cost += carbon_cost

        model.setObjective(objective_expr + total_cost, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 3000,
        'n_cols': 375,
        'density': 0.45,
        'max_coef': 112,
        'cover_fraction': 0.8,
        'n_facilities': 300,
        'n_demand_nodes': 5,
        'min_demand': 40,
        'max_demand': 1050,
        'min_transport_cost': 3.75,
        'max_transport_cost': 2500.0,
        'min_capacity': 900,
        'max_capacity': 1875,
        'min_facility_cost': 50000,
        'max_facility_cost': 200000,
        'min_processing_cost': 500,
        'max_processing_cost': 2000,
        'min_auxiliary_cost': 150,
        'max_auxiliary_cost': 600,
        'transportation_modes': ('land', 'sea', 'air'),
        'min_mode_capacity': 50,
        'max_mode_capacity': 750,
        'min_holding_cost': 5.0,
        'max_holding_cost': 100.0,
        'min_lead_time': 1.0,
        'max_lead_time': 30.0,
        'min_carbon': 0.73,
        'max_carbon': 100.0,
    }

    set_cover_problem = SetCoverWithSupplyChain(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")