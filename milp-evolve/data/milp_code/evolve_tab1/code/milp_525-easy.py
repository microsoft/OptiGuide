import random
import time
import numpy as np
import networkx as nx
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
        weights = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (self.max_range - self.min_range), 1),
                               weights + (self.max_range - self.min_range)]))

        elif self.scheme == 'strongly correlated':
            profits = weights + (self.max_range - self.min_range) / 10

        elif self.scheme == 'subset-sum':
            profits = weights

        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_facilities, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_facilities,
                                            0.6 * weights.sum() // self.number_of_facilities,
                                            self.number_of_facilities - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities}

        return res
        
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        
        number_of_items = len(weights)
        number_of_facilities = len(capacities)
        
        model = Model("FacilityLocation")
        var_x = {}
        var_z = {}

        # Decision variables: x[i][j] = 1 if item i is assigned to facility j
        for i in range(number_of_items):
            for j in range(number_of_facilities):
                var_x[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Decision variables: z[j] = 1 if facility j is open
        for j in range(number_of_facilities):
            var_z[j] = model.addVar(vtype="B", name=f"z_{j}")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_x[(i, j)] for i in range(number_of_items) for j in range(number_of_facilities))

        # Constraints: Each item can be in at most one facility
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_x[(i, j)] for j in range(number_of_facilities)) <= 1,
                f"ItemAssignment_{i}"
            )

        # Big M Constraints: Item i can only be assigned to facility j if facility j is open
        M = max(profits)  # Use the maximum profit as Big M
        for i in range(number_of_items):
            for j in range(number_of_facilities):
                model.addCons(var_x[(i, j)] <= var_z[j] * M, f"BigM_{i}_{j}")

        # Constraints: Total weight in each facility must not exceed its capacity
        for j in range(number_of_facilities):
            model.addCons(
                quicksum(weights[i] * var_x[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"FacilityCapacity_{j}"
            )

        # Constraints: At least one facility must be open
        model.addCons(
            quicksum(var_z[j] for j in range(number_of_facilities)) >= 1,
            "AtLeastOneFacility"
        )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 1000,
        'number_of_facilities': 5,
        'min_range': 5,
        'max_range': 60,
        'scheme': 'weakly correlated',
    }

    facility_location = FacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")