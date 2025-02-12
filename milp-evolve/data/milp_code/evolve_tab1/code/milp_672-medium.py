import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityPlacementOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_facility_locations = random.randint(self.min_facility_locations, self.max_facility_locations)
        num_neighborhoods = random.randint(self.min_neighborhoods, self.max_neighborhoods)
        
        # Cost and intensity matrices
        transportation_costs = np.random.randint(1, 100, size=(num_neighborhoods, num_facility_locations))
        construction_costs = np.random.randint(500, 1000, size=num_facility_locations)
        facility_budget = np.random.randint(10000, 20000)
        
        min_construction_levels = np.random.randint(1, 5, size=num_facility_locations)
        surplus_penalty = np.random.randint(10, 50)
        seasonal_variation = np.random.normal(0, 0.1, size=num_facility_locations)

        res = {
            'num_facility_locations': num_facility_locations,
            'num_neighborhoods': num_neighborhoods,
            'transportation_costs': transportation_costs,
            'construction_costs': construction_costs,
            'facility_budget': facility_budget,
            'min_construction_levels': min_construction_levels,
            'surplus_penalty': surplus_penalty,
            'seasonal_variation': seasonal_variation,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_facility_locations = instance['num_facility_locations']
        num_neighborhoods = instance['num_neighborhoods']
        transportation_costs = instance['transportation_costs']
        construction_costs = instance['construction_costs']
        facility_budget = instance['facility_budget']
        min_construction_levels = instance['min_construction_levels']
        surplus_penalty = instance['surplus_penalty']
        seasonal_variation = instance['seasonal_variation']
        
        model = Model("FacilityPlacementOptimization")
        
        # Variables
        facility = {j: model.addVar(vtype="B", name=f"facility_{j}") for j in range(num_facility_locations)}
        accessibility = {(i, j): model.addVar(vtype="B", name=f"accessibility_{i}_{j}") for i in range(num_neighborhoods) for j in range(num_facility_locations)}
        surplus = {j: model.addVar(vtype="C", name=f"surplus_{j}") for j in range(num_facility_locations)}
        
        # Objective function: Minimize total costs
        total_cost = quicksum(accessibility[i, j] * transportation_costs[i, j] for i in range(num_neighborhoods) for j in range(num_facility_locations)) + \
                     quicksum(facility[j] * construction_costs[j] for j in range(num_facility_locations)) + \
                     quicksum(surplus[j] * surplus_penalty for j in range(num_facility_locations))
        
        model.setObjective(total_cost, "minimize")
        
        # Constraints
        
        # Each neighborhood should be accessible to at least one facility
        for i in range(num_neighborhoods):
            model.addCons(quicksum(accessibility[i, j] for j in range(num_facility_locations)) >= 1, name=f"neighborhood_accessibility_{i}")
        
        # A facility can only provide access if it is built
        for j in range(num_facility_locations):
            for i in range(num_neighborhoods):
                model.addCons(accessibility[i, j] <= facility[j], name=f"facility_accessibility_{i}_{j}")
        
        # Budget constraint on the total construction cost
        model.addCons(quicksum(facility[j] * construction_costs[j] for j in range(num_facility_locations)) <= facility_budget, name="budget_constraint")
        
        # Minimum construction level if activated (Big M Formulation)
        for j in range(num_facility_locations):
            model.addCons(facility[j] >= min_construction_levels[j] * facility[j], name=f"min_construction_level_{j}")

        # Seasonal variations influencing construction capacities
        for j in range(num_facility_locations):
            slope = seasonal_variation[j]
            model.addCons(facility[j] <= construction_costs[j] * (1 + slope), name=f"seasonal_variation_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facility_locations': 10,
        'max_facility_locations': 600,
        'min_neighborhoods': 160,
        'max_neighborhoods': 420,
    }

    optimization = FacilityPlacementOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")