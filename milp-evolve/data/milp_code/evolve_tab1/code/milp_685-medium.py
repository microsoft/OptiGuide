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
    def normal_int(self, size, mean, std_dev, lower_bound, upper_bound):
        return np.clip(
            np.round(np.random.normal(mean, std_dev, size)), 
            lower_bound, 
            upper_bound
        ).astype(int)

    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2 +
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2
        )
        return costs

    def generate_instance(self):
        demands = self.normal_int(
            self.n_customers_emergency, 
            self.demand_mean, 
            self.demand_std, 
            self.demand_lower, 
            self.demand_upper
        )
        capacities = self.normal_int(
            self.n_shelters, 
            self.capacity_mean, 
            self.capacity_std, 
            self.capacity_lower, 
            self.capacity_upper
        )

        # Simplifying by removing fixed costs for medical equipment
        fixed_costs = (
            self.normal_int(
                self.n_shelters, 
                self.helipad_construction_cost_mean, 
                self.helipad_construction_cost_std, 
                self.helipad_construction_cost_lower, 
                self.helipad_construction_cost_upper
            ) * np.sqrt(capacities)
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
        serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers_emergency) for j in range(n_shelters)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_shelters[j] for j in range(n_shelters)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers_emergency) for j in range(n_shelters))

        model.setObjective(objective_expr, "minimize")

        # Constraints: each customer must be served
        for i in range(n_customers_emergency):
            model.addCons(quicksum(serve[i, j] for j in range(n_shelters)) >= 1, f"Serving_Customers_{i}")

        # Constraints: capacity limits at each shelter
        for j in range(n_shelters):
            model.addCons(
                quicksum(serve[i, j] * demands[i] for i in range(n_customers_emergency)) <= capacities[j] * open_shelters[j], 
                f"Shelter_Capacity_{j}"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers_emergency': 200,
        'n_shelters': 100,
        'demand_mean': 900,
        'demand_std': 2100,
        'demand_lower': 600,
        'demand_upper': 810,
        'capacity_mean': 1875,
        'capacity_std': 75,
        'capacity_lower': 280,
        'capacity_upper': 90,
        'helipad_construction_cost_mean': 1350,
        'helipad_construction_cost_std': 900,
        'helipad_construction_cost_lower': 2025,
        'helipad_construction_cost_upper': 2700,
        'ratio': 200.0,
    }

    resource_allocation = EmergencyShelterResourceAllocation(parameters, seed=seed)
    instance = resource_allocation.generate_instance()
    solve_status, solve_time = resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")