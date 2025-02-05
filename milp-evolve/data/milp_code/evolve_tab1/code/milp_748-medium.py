import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ComplexCombinatorialAuction:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.min_value >= 0 and self.max_value >= self.min_value
        assert self.add_item_prob >= 0 and self.add_item_prob <= 1

        # Generate correlated item values
        mean = (self.min_value + self.max_value) / 2
        cov = np.full((self.n_items, self.n_items), self.correlation) * (self.max_value - self.min_value) / 3
        np.fill_diagonal(cov, (self.max_value - self.min_value) ** 2 / 12)
        values = np.random.multivariate_normal(mean * np.ones(self.n_items), cov)

        # bids generation
        bids = []

        for _ in range(self.n_bids):
            private_interests = np.random.rand(self.n_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)

            initial_item = np.random.choice(self.n_items, p=private_interests / private_interests.sum())
            bundle_mask = np.zeros(self.n_items, dtype=bool)
            bundle_mask[initial_item] = True

            while np.random.rand() < self.add_item_prob and bundle_mask.sum() < self.n_items:
                next_item = np.random.choice(self.n_items, p=(private_interests * ~bundle_mask) / (private_interests * ~bundle_mask).sum())
                bundle_mask[next_item] = True

            bundle = np.nonzero(bundle_mask)[0]
            price = private_values[bundle].sum() + len(bundle) ** (1 + self.additivity) + np.random.exponential(scale=100)

            if price > 0:
                bids.append((list(bundle), price))

        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)
        
        budget = np.random.uniform(self.min_budget, self.max_budget)

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "budget": budget
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        budget = instance['budget']
        
        model = Model("ComplexCombinatorialAuction")
        
        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        expenditure_vars = {i: model.addVar(vtype="C", name=f"Expenditure_{i}") for i in range(len(bids))}
        
        # Objective: maximize the total price minus the penalty for budget usage
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))
        penalty_expr = quicksum(expenditure_vars[i] for i in range(len(bids))) / budget
        model.setObjective(objective_expr - penalty_expr, "maximize")
        
        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")
        
        # Budget constraint
        model.addCons(quicksum(expenditure_vars[i] for i in range(len(bids))) <= budget, "Budget")

        # Linking expenditure to bid selection
        for i, (bundle, price) in enumerate(bids):
            model.addCons(expenditure_vars[i] == price * bid_vars[i], f"Expenditure_{i}")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 2000,
        'n_bids': 2700,
        'min_value': 5,
        'max_value': 500,
        'value_deviation': 0.1,
        'additivity': 0.8,
        'add_item_prob': 0.24,
        'min_budget': 10000,
        'max_budget': 50000,
        'correlation': 0.73,
    }

    auction = ComplexCombinatorialAuction(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")