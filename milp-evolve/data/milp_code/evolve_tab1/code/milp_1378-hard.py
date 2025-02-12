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

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        num_pairs = self.number_of_pairs
        mutual_inclusion_pair = set()
        mutual_exclusion_pair = set()
        
        while len(mutual_inclusion_pair) < num_pairs:
            i, j = np.random.choice(self.number_of_items, 2, replace=False)
            mutual_inclusion_pair.add((i, j))
            
        while len(mutual_exclusion_pair) < num_pairs:
            i, j = np.random.choice(self.number_of_items, 2, replace=False)
            mutual_exclusion_pair.add((i, j))
            
        mutual_inclusion_pair = list(mutual_inclusion_pair)
        mutual_exclusion_pair = list(mutual_exclusion_pair)

        res = {'weights': weights,
               'profits': profits,
               'capacities': capacities,
               'mutual_inclusion_pair': mutual_inclusion_pair,
               'mutual_exclusion_pair': mutual_exclusion_pair}

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        mutual_inclusion_pair = instance['mutual_inclusion_pair']
        mutual_exclusion_pair = instance['mutual_exclusion_pair']
        
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
        
        # Mutual inclusion constraints
        for (i1, i2) in mutual_inclusion_pair:
            for j in range(number_of_knapsacks):
                model.addCons(
                    var_names[(i1, j)] - var_names[(i2, j)] == 0,
                    f"MutualInclusion_{i1}_{i2}_{j}"
                )
                
        # Mutual exclusion constraints
        for (i1, i2) in mutual_exclusion_pair:
            for j in range(number_of_knapsacks):
                model.addCons(
                    var_names[(i1, j)] + var_names[(i2, j)] <= 1,
                    f"MutualExclusion_{i1}_{i2}_{j}"
                )
                        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 2000,
        'number_of_knapsacks': 5,
        'min_range': 7,
        'max_range': 15,
        'scheme': 'weakly correlated',
        'number_of_pairs': 3,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")