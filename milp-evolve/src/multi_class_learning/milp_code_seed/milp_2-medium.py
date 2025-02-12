'''
Reimplemented and formulated from the mknapsack implementation from https://github.com/lascavana/rl2branch/blob/master/01_generate_instances.py
'''

import random
import time
import numpy as np
import networkx as nx
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
        weights = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (self.max_range-self.min_range), 1),
                               weights + (self.max_range-self.min_range)]))

        elif self.scheme == 'strongly correlated':
            profits = weights + (self.max_range - self.min_range) / 10

        elif self.scheme == 'subset-sum':
            profits = weights

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

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 150,
        'number_of_knapsacks': 8,
        'min_range': 10,
        'max_range': 20,
        'scheme': 'weakly correlated'
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
