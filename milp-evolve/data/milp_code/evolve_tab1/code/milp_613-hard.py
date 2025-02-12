import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

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
        neighborhood_demand = np.random.randint(100, 1000, size=num_neighborhoods)

        # Facility capacities
        facility_capacity = np.random.randint(1000, 3000, size=num_facility_locations)

        res = {
            'num_facility_locations': num_facility_locations,
            'num_neighborhoods': num_neighborhoods,
            'transportation_costs': transportation_costs,
            'construction_costs': construction_costs,
            'facility_budget': facility_budget,
            'neighborhood_demand': neighborhood_demand,
            'facility_capacity': facility_capacity,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_facility_locations = instance['num_facility_locations']
        num_neighborhoods = instance['num_neighborhoods']
        transportation_costs = instance['transportation_costs']
        construction_costs = instance['construction_costs']
        facility_budget = instance['facility_budget']
        neighborhood_demand = instance['neighborhood_demand']
        facility_capacity = instance['facility_capacity']

        model = Model("FacilityPlacementOptimization")

        # Variables
        facility = {j: model.addVar(vtype="B", name=f"facility_{j}") for j in range(num_facility_locations)}
        accessibility = {(i, j): model.addVar(vtype="B", name=f"accessibility_{i}_{j}") for i in range(num_neighborhoods) for j in range(num_facility_locations)}
        intensity = {j: model.addVar(vtype="I", name=f"intensity_{j}") for j in range(num_facility_locations)}

        # Objective function: Minimize total costs
        total_cost = quicksum(accessibility[i, j] * transportation_costs[i, j] for i in range(num_neighborhoods) for j in range(num_facility_locations)) + \
                     quicksum(facility[j] * construction_costs[j] for j in range(num_facility_locations))

        model.setObjective(total_cost, "minimize")

        # Constraints

        # Each neighborhood should be accessible to at least one facility
        for i in range(num_neighborhoods):
            model.addCons(quicksum(accessibility[i, j] for j in range(num_facility_locations)) >= 1, name=f"neighborhood_accessibility_{i}")

        # A facility can only provide access if it is built
        for j in range(num_facility_locations):
            for i in range(num_neighborhoods):
                model.addCons(accessibility[i, j] <= facility[j], name=f"facility_accessibility_{i}_{j}")

        # Facility capacity constraints
        for j in range(num_facility_locations):
            model.addCons(quicksum(accessibility[i, j] * neighborhood_demand[i] for i in range(num_neighborhoods)) <= facility_capacity[j], name=f"facility_capacity_{j}")

        # Budget constraint on the total construction cost
        model.addCons(quicksum(facility[j] * construction_costs[j] for j in range(num_facility_locations)) <= facility_budget, name="budget_constraint")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facility_locations': 20,
        'max_facility_locations': 100,
        'min_neighborhoods': 30,
        'max_neighborhoods': 140,
    }

    optimization = FacilityPlacementOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")