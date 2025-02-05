import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        locations = np.random.rand(self.number_of_facilities, 2)
        demands = np.random.randint(self.min_demand, self.max_demand, self.number_of_customers)
        capacities = np.random.randint(self.min_capacity, self.max_capacity, self.number_of_facilities)
        opening_costs = np.random.randint(self.min_opening_cost, self.max_opening_cost, self.number_of_facilities)

        transportation_costs = np.zeros((self.number_of_facilities, self.number_of_customers))
        for f in range(self.number_of_facilities):
            for c in range(self.number_of_customers):
                transportation_costs[f, c] = np.linalg.norm(locations[f] - np.random.rand(2))

        res = {
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'transportation_costs': transportation_costs
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        transportation_costs = instance['transportation_costs']

        number_of_customers = len(demands)
        number_of_facilities = len(capacities)
        
        model = Model("FacilityLocation")
        facility_open = {}
        allocate = {}

        # Decision variables
        for f in range(number_of_facilities):
            facility_open[f] = model.addVar(vtype="B", name=f"FacilityOpen_{f}")
            for c in range(number_of_customers):
                allocate[(f, c)] = model.addVar(vtype="B", name=f"Allocate_{f}_{c}")

        # Objective: Minimize opening and transportation costs
        objective_expr = quicksum(opening_costs[f] * facility_open[f] for f in range(number_of_facilities)) + \
                         quicksum(transportation_costs[f, c] * allocate[(f, c)] for f in range(number_of_facilities) for c in range(number_of_customers))
        model.setObjective(objective_expr, "minimize")

        # Constraints: Each customer is assigned to exactly one facility
        for c in range(number_of_customers):
            model.addCons(
                quicksum(allocate[(f, c)] for f in range(number_of_facilities)) == 1,
                f"CustomerAssignment_{c}"
            )

        # Constraints: Total demand at each facility should not exceed its capacity
        for f in range(number_of_facilities):
            model.addCons(
                quicksum(demands[c] * allocate[(f, c)] for c in range(number_of_customers)) <= capacities[f] * facility_open[f],
                f"FacilityCapacity_{f}"
            )

        # Constraints: A customer can only be allocated to an open facility
        for f in range(number_of_facilities):
            for c in range(number_of_customers):
                model.addCons(
                    allocate[(f, c)] <= facility_open[f],
                    f"AllocateToOpenFacility_{f}_{c}"
                )

        # Solve the model
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_customers': 50,
        'number_of_facilities': 50,
        'min_demand': 50,
        'max_demand': 250,
        'min_capacity': 10,
        'max_capacity': 1500,
        'min_opening_cost': 1000,
        'max_opening_cost': 5000,
    }

    facility_location = FacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")