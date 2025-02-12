import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedCombinatorialAuction:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.min_value >= 0 and self.max_value >= self.min_value
        assert self.add_item_prob >= 0 and self.add_item_prob <= 1

        # Common item values (resale price)
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)

        bids = []

        # Create bids, one bidder at a time
        while len(bids) < self.n_bids:
            private_interests = np.random.rand(self.n_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)

            # Generate initial bundle (simplify by choosing a random item first)
            item = np.random.choice(self.n_items)
            bundle_mask = np.full(self.n_items, 0)
            bundle_mask[item] = 1

            # Add additional items based on a probability
            while np.random.rand() < self.add_item_prob:
                if bundle_mask.sum() == self.n_items:
                    break
                item = np.random.choice(np.where(bundle_mask == 0)[0])
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]

            # Simplified bundle price calculation
            price = private_values[bundle].sum() * 0.5

            # Create the bid
            bids.append((list(bundle), price))

        bids_per_item = [[] for item in range(self.n_items)]
        for i, (bundle, price) in enumerate(bids):
            for item in bundle:
                bids_per_item[item].append(i)

        return {
            "bids": bids,
            "bids_per_item": bids_per_item
        }

    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']

        model = Model("SimplifiedCombinatorialAuction")
        
        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}

        # Objective: maximize the total price
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))

        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 800,
        'n_bids': 720,
        'min_value': 1260,
        'max_value': 2250,
        'value_deviation': 0.69,
        'add_item_prob': 0.77,
    }

    auction = SimplifiedCombinatorialAuction(parameters, seed)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")