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

        # Generate time windows for each item
        time_windows = [(0, 0)] * self.number_of_items
        for i in range(self.number_of_items):
            start_time = np.random.randint(self.min_time, self.max_time)
            end_time = start_time + np.random.randint(self.min_duration, self.max_duration)
            time_windows[i] = (start_time, end_time)

        # Resource requirements 
        resource_requirements = np.random.randint(self.min_requirement, self.max_requirement, size=(self.number_of_items, self.n_resources))

        res = {'weights': weights,
               'profits': profits,
               'capacities': capacities,
               'time_windows': time_windows,
               'resource_requirements': resource_requirements}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        time_windows = instance['time_windows']
        resource_requirements = instance['resource_requirements']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsack")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )

        # New Constraints: Ensure items are placed within their time windows
        for i in range(number_of_items):
            start_time, end_time = time_windows[i]
            for j in range(number_of_knapsacks):
                model.addCons(
                    start_time * var_names[(i, j)] <= end_time,
                    f"TimeWindow_{i}_{j}"
                )

        # New Constraints: Ensure resource requirements do not exceed available resources
        for r in range(self.n_resources):
            for j in range(number_of_knapsacks):
                model.addCons(
                    quicksum(resource_requirements[i][r] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j] * self.resource_factors[r],
                    f"ResourceCapacity_{r}_{j}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 3000,
        'number_of_knapsacks': 2,
        'min_range': 3,
        'max_range': 1500,
        'weight_mean': 50,
        'weight_std': 60,
        'profit_mean_shift': 15,
        'profit_std': 5,
        'n_resources': 0,
        'resource_factors': (0.12, 0.17),
        'min_time': 0,
        'max_time': 2,
        'min_duration': 0,
        'max_duration': 7,
        'min_requirement': 7,
        'max_requirement': 1000,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")