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

        # Logical Condition 1: If item A is included in any knapsack, item B must also be included.
        item_A, item_B = 0, 1  # Example items
        model.addCons(
            quicksum(var_names[(item_A, j)] for j in range(number_of_knapsacks)) <= quicksum(var_names[(item_B, j)] for j in range(number_of_knapsacks)),
            "LogicalCondition_1"
        )

        # Logical Condition 2: Items C and D must be packed in the same knapsack.
        item_C, item_D = 2, 3  # Example items
        for j in range(number_of_knapsacks):
            model.addCons(
                var_names[(item_C, j)] == var_names[(item_D, j)],
                f"LogicalCondition_2_{j}"
            )

        # Logical Condition 3: Item E and F cannot be packed in the same knapsack.
        item_E, item_F = 4, 5  # Example items
        for j in range(number_of_knapsacks):
            model.addCons(
                var_names[(item_E, j)] + var_names[(item_F, j)] <= 1,
                f"LogicalCondition_3_{j}"
            )

        # Logical Condition 4: At least a minimum number of items must be packed in each knapsack for it to be considered utilized.
        min_items_per_knapsack = 2
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(var_names[(i, j)] for i in range(number_of_items)) >= min_items_per_knapsack * var_names[(i, j)],
                f"LogicalCondition_4_{j}"
            )

        ### New Logical Conditions using Convex Hull Formulation ###
        
        # Logical Condition 5: If item G is packed, item H must also be packed.
        item_G, item_H = 6, 7  # Example items
        model.addCons(
            quicksum(var_names[(item_G, j)] for j in range(number_of_knapsacks)) <= quicksum(var_names[(item_H, j)] for j in range(number_of_knapsacks)),
            "LogicalCondition_5"
        )

        # Logical Condition 6: Items I and J must be together but not with item K.
        item_I, item_J, item_K = 8, 9, 10  # Example items
        for j in range(number_of_knapsacks):
            model.addCons(
                var_names[(item_I, j)] == var_names[(item_J, j)],
                f"LogicalCondition_6_1_{j}"
            )
            model.addCons(
                var_names[(item_I, j)] + var_names[(item_K, j)] <= 1,
                f"LogicalCondition_6_2_{j}"
            )

        # Secondary Objective: Minimize the number of knapsacks used
        z = model.addVar(vtype="B", name="z")
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(var_names[(i, j)] for i in range(number_of_items)) >= z,
                f"KnapsackUsed_{j}"
            )

        model.setObjective(objective_expr - z, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 180,
        'number_of_knapsacks': 70,
        'min_range': 3,
        'max_range': 16,
        'scheme': 'weakly correlated',
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")