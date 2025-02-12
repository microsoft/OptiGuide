import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MultipleKnapsack:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        # Simplified data generation: weights and profits are now highly correlated
        weights = np.random.normal(loc=self.weight_mean, scale=self.weight_std, size=self.number_of_items).astype(int)
        profits = weights + np.random.normal(loc=self.profit_mean_shift, scale=self.profit_std, size=self.number_of_items).astype(int)

        # Ensure non-negative values
        weights = np.clip(weights, self.min_range, self.max_range)
        profits = np.clip(profits, self.min_range, self.max_range)

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
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
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsack")
        var_names = {}
        z = {}

        M = max(weights.sum(), capacities.sum())

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
            z[i] = model.addVar(vtype="B", name=f"z_{i}")

        # Objective: Maximize total profit with slightly altered structure for complexity
        objective_expr = quicksum((profits[i] * (j+1)) * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= z[i],
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )
        
        # Additional constraints for diversity: enforce minimum utilization of each knapsack
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) >= 0.1 * capacities[j],
                f"KnapsackMinUtilization_{j}"
            )

        # Big M Constraints: Ensure z[i] logically connects to x[i][j]
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                model.addCons(var_names[(i, j)] <= z[i], f"BigM_constraint_1_{i}_{j}")  # If x[i][j] is 1, z[i] must be 1
                model.addCons(var_names[(i, j)] >= z[i] - (1 - var_names[(i, j)]), f"BigM_constraint_2_{i}_{j}")  # If z[i] is 1, at least one x[i][j] must be 1

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 150,
        'number_of_knapsacks': 5,
        'min_range': 5,
        'max_range': 150,
        'weight_mean': 200,
        'weight_std': 50,
        'profit_mean_shift': 50,
        'profit_std': 6,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")