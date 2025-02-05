import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EmergencyShelterResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2 +
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers_emergency, self.demand_interval)
        capacities = self.randint(self.n_shelters, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_shelters, self.helipad_construction_cost_interval) * np.sqrt(capacities) +
            self.randint(self.n_shelters, self.medical_equipment_cost_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        
        n_customers_emergency = len(demands)
        n_shelters = len(capacities)
        
        model = Model("EmergencyShelterResourceAllocation")
        
        # Decision variables
        open_shelters = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_shelters)}
        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers_emergency) for j in range(n_shelters)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers_emergency) for j in range(n_shelters)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_shelters[j] for j in range(n_shelters)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers_emergency) for j in range(n_shelters))

        model.setObjective(objective_expr, "minimize")

        # Constraints: each depraved area must be served by at least one shelter
        for i in range(n_customers_emergency):
            model.addCons(quicksum(serve[i, j] for j in range(n_shelters)) >= 1, f"ServingDepravedAreas_{i}")

        # Constraints: capacity limits at each shelter
        for j in range(n_shelters):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers_emergency)) <= capacities[j] * open_shelters[j], f"ShelterCapacity_{j}")
        
        # General constraint on the total number of shelters to be opened
        model.addCons(quicksum(open_shelters[j] for j in range(n_shelters)) <= self.shelter_limit, "NumberOfShelters")

        for i in range(n_customers_emergency):
            for j in range(n_shelters):
                model.addCons(serve[i, j] <= open_shelters[j], f"Tightening_{i}_{j}")

        # Removed additional constraints here

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers_emergency': 50,
        'n_shelters': 50,
        'demand_interval': (15, 108),
        'capacity_interval': (7, 120),
        'helipad_construction_cost_interval': (900, 2700),
        'medical_equipment_cost_interval': (150, 300),
        'ratio': 5.0,
        'continuous_assignment': 0,
        'shelter_limit': 200,
    }

    resource_allocation = EmergencyShelterResourceAllocation(parameters, seed=seed)
    instance = resource_allocation.generate_instance()
    solve_status, solve_time = resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")