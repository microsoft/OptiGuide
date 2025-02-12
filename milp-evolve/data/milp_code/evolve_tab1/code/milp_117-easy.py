import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityCost_Minimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_distance_matrix(self, num_facilities, num_demands):
        return np.random.randint(self.min_distance, self.max_distance + 1, size=(num_facilities, num_demands))

    def generate_instance(self):
        num_facilities = np.random.randint(self.min_facilities, self.max_facilities + 1)
        num_demands = np.random.randint(self.min_demands, self.max_demands + 1)
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, size=num_facilities)
        demand_quantities = np.random.randint(self.min_demand_quantity, self.max_demand_quantity + 1, size=num_demands)
        capacity_limits = np.random.randint(self.min_capacity_limit, self.max_capacity_limit + 1, size=num_facilities)
        distance_matrix = self.generate_distance_matrix(num_facilities, num_demands)

        res = {
            'num_facilities': num_facilities,
            'num_demands': num_demands,
            'facility_costs': facility_costs,
            'demand_quantities': demand_quantities,
            'capacity_limits': capacity_limits,
            'distance_matrix': distance_matrix,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_facilities = instance['num_facilities']
        num_demands = instance['num_demands']
        facility_costs = instance['facility_costs']
        demand_quantities = instance['demand_quantities']
        capacity_limits = instance['capacity_limits']
        distance_matrix = instance['distance_matrix']

        model = Model("FacilityCost_Minimization")

        # Create variables
        facilities = {i: model.addVar(vtype="B", name=f"F_{i}") for i in range(num_facilities)}
        assignments = {(i, j): model.addVar(vtype="B", name=f"A_{i}_{j}") for i in range(num_facilities) for j in range(num_demands)}

        # Objective function
        facility_cost_term = quicksum(facility_costs[i] * facilities[i] for i in range(num_facilities))
        transportation_cost_term = quicksum(distance_matrix[i, j] * assignments[i, j] for i in range(num_facilities) for j in range(num_demands))

        model.setObjective(facility_cost_term + transportation_cost_term, "minimize")

        # Constraints
        for j in range(num_demands):
            model.addCons(quicksum(assignments[i, j] for i in range(num_facilities)) == 1, name=f"Demand_{j}")

        for i in range(num_facilities):
            model.addCons(quicksum(demand_quantities[j] * assignments[i, j] for j in range(num_demands)) <= capacity_limits[i] * facilities[i], name=f"Capacity_{i}")

        for i in range(num_facilities):
            for j in range(num_demands):
                model.addCons(assignments[i, j] <= facilities[i], name=f"Assignment_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facilities': 10,
        'max_facilities': 30,
        'min_demands': 50,
        'max_demands': 100,
        'min_facility_cost': 200,
        'max_facility_cost': 500,
        'min_demand_quantity': 10,
        'max_demand_quantity': 50,
        'min_capacity_limit': 100,
        'max_capacity_limit': 300,
        'min_distance': 5,
        'max_distance': 50,
    }

    facility_cost_minimization = FacilityCost_Minimization(parameters, seed=seed)
    instance = facility_cost_minimization.generate_instance()
    solve_status, solve_time = facility_cost_minimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")