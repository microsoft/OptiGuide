import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class ComplexSetCoverFacilityLocation:
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
        indices = np.random.choice(self.n_cols, size=nnzrs)  # random column indexes
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  # force at least 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)

        # for each column, sample random rows
        indices[:self.n_rows] = np.random.permutation(self.n_rows)  # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= self.n_rows:
                indices[i:i + n] = np.random.choice(self.n_rows, size=n, replace=False)
            # partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i + n] = np.random.choice(remaining_rows, size=i + n - self.n_rows, replace=False)
            i += n
            indptr.append(i)

        # objective coefficients for set cover
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # Additional data for activation costs (randomly pick some columns as crucial)
        crucial_sets = np.random.choice(self.n_cols, self.n_crucial, replace=False)
        activation_cost = np.random.randint(self.activation_cost_low, self.activation_cost_high, size=self.n_crucial)
        
        # Failure probabilities
        failure_probabilities = np.random.uniform(self.failure_probability_low, self.failure_probability_high, self.n_cols)
        penalty_costs = np.random.randint(self.penalty_cost_low, self.penalty_cost_high, size=self.n_cols)
        
        # New data for facility location problem
        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost, (self.n_facilities, self.n_cols))
        capacities = np.random.randint(self.min_capacity, self.max_capacity, self.n_facilities)
        traffic_congestion = np.random.uniform(1, 1.5, (self.n_facilities, self.n_cols))
        maintenance_schedules = np.random.choice([0, 1], (self.n_facilities, self.n_time_slots), p=[0.9, 0.1])
        electricity_prices = np.random.uniform(0.1, 0.5, self.n_time_slots)
        carbon_emissions_factors = np.random.uniform(0.05, 0.3, self.n_facilities)
        storage_costs = np.random.uniform(15, 50, self.n_facilities)

        res = {
            'c': c,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'crucial_sets': crucial_sets,
            'activation_cost': activation_cost,
            'failure_probabilities': failure_probabilities,
            'penalty_costs': penalty_costs,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs,
            'capacities': capacities,
            'traffic_congestion': traffic_congestion,
            'maintenance_schedules': maintenance_schedules,
            'electricity_prices': electricity_prices,
            'carbon_emissions_factors': carbon_emissions_factors,
            'storage_costs': storage_costs
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        crucial_sets = instance['crucial_sets']
        activation_cost = instance['activation_cost']
        failure_probabilities = instance['failure_probabilities']
        penalty_costs = instance['penalty_costs']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        traffic_congestion = instance['traffic_congestion']
        maintenance_schedules = instance['maintenance_schedules']
        electricity_prices = instance['electricity_prices']
        carbon_emissions_factors = instance['carbon_emissions_factors']
        storage_costs = instance['storage_costs']

        model = Model("ComplexSetCoverFacilityLocation")
        var_names = {}
        activate_crucial = {}
        fail_var_names = {}
        facility_vars = {}
        allocation_vars = {}
        time_slot_vars = {}
        energy_vars = {}
        storage_vars = {}

        # Create variables and set objective for classic set cover
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])
            fail_var_names[j] = model.addVar(vtype="B", name=f"f_{j}")

        # Additional variables for crucial sets activation
        for idx, j in enumerate(crucial_sets):
            activate_crucial[j] = model.addVar(vtype="B", name=f"y_{j}", obj=activation_cost[idx])

        # Facility location variables
        for f in range(self.n_facilities):
            facility_vars[f] = model.addVar(vtype="B", name=f"Facility_{f}", obj=fixed_costs[f])
            storage_vars[f] = model.addVar(vtype="C", name=f"Storage_{f}", lb=0, obj=storage_costs[f])
            energy_vars[f] = model.addVar(vtype="C", name=f"Energy_{f}", lb=0)

            for j in range(self.n_cols):
                allocation_vars[(f, j)] = model.addVar(vtype="B", name=f"Facility_{f}_Column_{j}")

        # Maintenance variables
        for f in range(self.n_facilities):
            for t in range(self.n_time_slots):
                time_slot_vars[(f, t)] = model.addVar(vtype="B", name=f"Facility_{f}_TimeSlot_{t}")

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] - fail_var_names[j] for j in cols) >= 1, f"c_{row}")

        # Ensure prioritized sets (crucial sets) have higher coverage conditions
        for j in crucial_sets:
            rows_impacting_j = np.where(indices_csr == j)[0]
            for row in rows_impacting_j:
                model.addCons(var_names[j] >= activate_crucial[j], f"crucial_coverage_row_{row}_set_{j}")

        # Facility capacity and assignment constraints using Convex Hull Formulation
        for f in range(self.n_facilities):
            capacity_expr = quicksum(allocation_vars[(f, j)] for j in range(self.n_cols))
            model.addCons(capacity_expr <= capacities[f], f"Facility_{f}_Capacity")
            for j in range(self.n_cols):
                model.addCons(allocation_vars[(f, j)] <= facility_vars[f], f"Facility_{f}_Alloc_{j}")
                model.addCons(allocation_vars[(f, j)] >= facility_vars[f] + var_names[j] - 1, f"ConvexAlloc_{f}_{j}")

            # Energy consumption constraint
            model.addCons(energy_vars[f] == quicksum(time_slot_vars[(f, t)] * electricity_prices[t] for t in range(self.n_time_slots)), f"Energy_Consumption_{f}")

        # Storage constraints
        for f in range(self.n_facilities):
            # Ensure that storage usage is within the capacity limits
            model.addCons(storage_vars[f] <= capacities[f], f"Storage_Capacity_{f}")

        # Maintenance constraints
        for f in range(self.n_facilities):
            for t in range(self.n_time_slots):
                model.addCons(time_slot_vars[(f, t)] <= facility_vars[f], f"Maintenance_Facility_{f}_TimeSlot_{t}")
                model.addCons(time_slot_vars[(f, t)] <= (1 - maintenance_schedules[f, t]), f"Maintenance_Scheduled_Facility_{f}_TimeSlot_{t}")

        # Objective: Minimize total cost including penalties for failures, fixed costs, transport costs, carbon emissions, storage costs, and energy consumption costs
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + \
                         quicksum(activate_crucial[j] * activation_cost[idx] for idx, j in enumerate(crucial_sets)) + \
                         quicksum(fail_var_names[j] * penalty_costs[j] for j in range(self.n_cols)) + \
                         quicksum(fixed_costs[f] * facility_vars[f] for f in range(self.n_facilities)) + \
                         quicksum(transport_costs[f][j] * allocation_vars[(f, j)] * traffic_congestion[f][j] for f in range(self.n_facilities) for j in range(self.n_cols)) + \
                         quicksum(carbon_emissions_factors[f] * energy_vars[f] for f in range(self.n_facilities)) + \
                         quicksum(storage_costs[f] * storage_vars[f] for f in range(self.n_facilities))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 562,
        'n_cols': 375,
        'density': 0.31,
        'max_coef': 10,
        'n_crucial': 33,
        'activation_cost_low': 54,
        'activation_cost_high': 1500,
        'failure_probability_low': 0.66,
        'failure_probability_high': 0.24,
        'penalty_cost_low': 75,
        'penalty_cost_high': 112,
        'n_facilities': 10,
        'min_fixed_cost': 13,
        'max_fixed_cost': 790,
        'min_transport_cost': 81,
        'max_transport_cost': 343,
        'min_capacity': 71,
        'max_capacity': 759,
        'n_time_slots': 24  # Updated to include more time slots for daily schedule
    }

    set_cover_problem = ComplexSetCoverFacilityLocation(parameters, seed=seed)
    instance = set_cover_problem.generate_instance()
    solve_status, solve_time = set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")