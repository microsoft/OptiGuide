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
        if self.dynamic_range:
            range_factor = np.random.uniform(0.5, 2.0)
            min_range = int(self.base_range * range_factor)
            max_range = int(self.base_range * range_factor * 2)
        else:
            min_range = self.min_range
            max_range = self.max_range

        weights = np.random.randint(min_range, max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(min_range, max_range, self.number_of_items)

        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (max_range-min_range), 1),
                    weights + (max_range-min_range)]))

        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities}

        ### new instance data code starts here
        pair_prohibit = [(i, (i+1) % self.number_of_items) for i in range(0, self.number_of_items, 3)]
        group_dependency = [list(range(i, min(i + 5, self.number_of_items))) for i in range(0, self.number_of_items, 5)]
        
        res['pair_prohibit'] = pair_prohibit
        res['group_dependency'] = group_dependency
        ### new instance data code ends here

        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        pair_prohibit = instance['pair_prohibit']
        group_dependency = instance['group_dependency']
        
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
        model.setObjective(objective_expr, "maximize")

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

        ### new constraints and variables and objective code starts here

        # Define a large M value
        M = max(capacities)

        # Pair prohibition constraint using Big M
        for i, k in pair_prohibit:
            for j in range(number_of_knapsacks):
                y_ikj = model.addVar(vtype="B", name=f"y_{i}_{k}_{j}")
                model.addCons(var_names[(i, j)] + var_names[(k, j)] <= 1 + M * (1 - y_ikj))
        
        # Group dependency constraint
        for group in group_dependency:
            for i in group:
                for k in group:
                    if i != k:
                        for j in range(number_of_knapsacks):
                            model.addCons(var_names[(i, j)] == var_names[(k, j)])

        ### new constraints and variables and objective code ends here
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 675,
        'number_of_knapsacks': 45,
        'min_range': 80,
        'max_range': 750,
        'base_range': 1800,
        'dynamic_range': 0,
        'scheme': 'weakly correlated',
    }
    
    ### new parameter code starts here
    M = 100  # Big M value for constraints
    ### new parameter code ends here

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")