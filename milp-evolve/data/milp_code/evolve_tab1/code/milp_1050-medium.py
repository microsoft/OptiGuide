import random
import time
import numpy as np
import networkx as nx
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

        # Bids generation
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.number_of_items)
        bids = []
        for _ in range(self.number_of_bids):
            private_interests = np.random.rand(self.number_of_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)

            initial_item = np.random.choice(self.number_of_items, p=private_interests / private_interests.sum())
            bundle_mask = np.zeros(self.number_of_items, dtype=bool)
            bundle_mask[initial_item] = True

            while np.random.rand() < self.add_item_prob and bundle_mask.sum() < self.number_of_items:
                next_item = np.random.choice(self.number_of_items, p=(private_interests * ~bundle_mask) / (private_interests * ~bundle_mask).sum())
                bundle_mask[next_item] = True

            bundle = np.nonzero(bundle_mask)[0]
            price = private_values[bundle].sum() + len(bundle) ** (1 + self.additivity)

            if price > 0:
                bids.append((list(bundle), price))

        # Capacity constraints
        transportation_capacity = np.random.randint(1, 11, size=self.number_of_items)

        bids_per_item = [[] for _ in range(self.number_of_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        res = {
            'weights': weights, 
            'profits': profits, 
            'capacities': capacities,
            'bids': bids,
            'bids_per_item': bids_per_item, 
            'transportation_capacity': transportation_capacity
        }

        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        transportation_capacity = instance['transportation_capacity']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        number_of_bids = len(bids)
        
        model = Model("MultipleKnapsackWithAuction")
        var_names = {}
        bid_vars = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Decision variables: b[k] = 1 if bid k is accepted
        for k in range(number_of_bids):
            bid_vars[k] = model.addVar(vtype="B", name=f"bid_{k}")

        # Objective: Maximize total profit and auction revenue
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) + \
                         quicksum(bids[k][1] * bid_vars[k] for k in range(number_of_bids))

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

        # Constraints: Each item can be at most one of a winning bid or placed in a knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) + 
                quicksum(bid_vars[k] for k in bids_per_item[i]) <= 1,
                f"ItemKnapsackOrBid_{i}"
            )

        # Constraints: Each item can be in at most one bundle of winning bids subject to transportation capacity
        for i in range(number_of_items):
            model.addCons(
                quicksum(bid_vars[k] for k in bids_per_item[i]) <= transportation_capacity[i],
                f"ItemTransportation_{i}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 400,
        'number_of_knapsacks': 20,
        'min_range': 10,
        'max_range': 600,
        'scheme': 'weakly correlated',
        'number_of_bids': 720,
        'min_value': 225,
        'max_value': 2700,
        'value_deviation': 0.74,
        'additivity': 0.74,
        'add_item_prob': 0.77,
    }
 
    knapsack_auction = MultipleKnapsackWithAuction(parameters, seed=seed)
    instance = knapsack_auction.generate_instance()
    solve_status, solve_time = knapsack_auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")