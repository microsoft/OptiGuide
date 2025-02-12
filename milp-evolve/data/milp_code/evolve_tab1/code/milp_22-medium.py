import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MultipleKnapsackWithAuction:
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

        mutual_exclusivity_pairs = [(random.randint(0, self.number_of_items - 1), random.randint(0, self.number_of_items - 1)) for _ in range(self.number_of_exclusive_pairs)]
        bids = [(random.sample(range(self.number_of_items), random.randint(1, self.max_bundle_size)), random.randint(self.min_profit_bid, self.max_profit_bid)) for _ in range(self.number_of_bids)]
        
        return {
            'weights': weights, 
            'profits': profits, 
            'capacities': capacities,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
            'bids': bids
        }
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        bids = instance['bids']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsackWithAuction")
        var_names = {}
        bundle_vars = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Bundle decision variables
        for b_idx, (bundle, price) in enumerate(bids):
            bundle_vars[b_idx] = model.addVar(vtype="B", name=f"bundle_bid_{b_idx}")

        # Objective: Maximize total profit (including bundles)
        objective_expr = (quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) +
                          quicksum(bids[b_idx][1] * bundle_vars[b_idx] for b_idx in range(len(bids))))

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

        # Constraints: Mutual exclusivity of items
        for i1, i2 in mutual_exclusivity_pairs:
            for j in range(number_of_knapsacks):
                model.addCons(var_names[(i1, j)] + var_names[(i2, j)] <= 1, f"MutualExclusivity_{i1}_{i2}_{j}")

        # Constraints: Bundle item inclusion
        for b_idx, (bundle, _) in enumerate(bids):
            for i in bundle:
                model.addCons(sum(var_names[(i, j)] for j in range(number_of_knapsacks)) >= bundle_vars[b_idx], f"Bundle_{b_idx}_{i}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 150,
        'number_of_knapsacks': 10,
        'min_range': 2,
        'max_range': 300,
        'scheme': 'weakly correlated',
        'number_of_exclusive_pairs': 37,
        'number_of_bids': 20,
        'max_bundle_size': 5,
        'min_profit_bid': 300,
        'max_profit_bid': 900,
    }

    knapsack = MultipleKnapsackWithAuction(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")