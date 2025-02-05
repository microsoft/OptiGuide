import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MultipleKnapsackWithBids:
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

        n_bids = int(self.number_of_items / 3)
        bids = []
        for _ in range(n_bids):
            # Randomly pick items to form a bid
            selected_items = np.random.choice(self.number_of_items, self.items_per_bid, replace=False)
            bid_profit = profits[selected_items].sum()
            bids.append((selected_items.tolist(), bid_profit))
        
        bid_pairs = []
        for _ in range(n_bids // 4):
            bid_pairs.append((
                random.randint(0, n_bids - 1),
                random.randint(0, n_bids - 1)
            ))

        res = {
            'weights': weights,
            'profits': profits,
            'capacities': capacities,
            'bids': bids,
            'mutual_exclusivity_pairs': bid_pairs
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        bids = instance['bids']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsackWithBids")
        var_names = {}
        x_bids = {i: model.addVar(vtype="B", name=f"x_bid_{i}") for i in range(len(bids))}
        waste_vars = {i: model.addVar(vtype="C", name=f"waste_{i}") for i in range(len(bids))}
        
        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) - quicksum(waste_vars[i] for i in range(len(bids)))

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

        # Constraints: Handle the newly introduced bids
        for i, (selected_items, bid_profit) in enumerate(bids):
            model.addCons(
                quicksum(var_names[(item, k)] for item in selected_items for k in range(number_of_knapsacks)) <= len(selected_items) * x_bids[i]
            )
        
        # Constraints: Mutual exclusivity among certain bids
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(x_bids[bid1] + x_bids[bid2] <= 1, name=f"Exclusive_bids_{bid1}_{bid2}")

        # Constraints: Waste penalty
        for i in range(len(bids)):
            model.addCons(waste_vars[i] >= 0, f"Waste_LB_{i}")
            model.addCons(waste_vars[i] >= self.waste_factor * (1 - x_bids[i]), f"Waste_Link_{i}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 1000,
        'number_of_knapsacks': 10,
        'min_range': 2,
        'max_range': 90,
        'scheme': 'weakly correlated',
        'items_per_bid': 2,
        'waste_factor': 0.24,
    }
    knapsack_with_bids = MultipleKnapsackWithBids(parameters, seed=seed)
    instance = knapsack_with_bids.generate_instance()
    solve_status, solve_time = knapsack_with_bids.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")