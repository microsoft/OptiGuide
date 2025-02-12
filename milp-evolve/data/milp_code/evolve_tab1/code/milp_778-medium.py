import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CombinatorialAuction:
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

        def choose_next_item(bundle_mask, interests, compats):
            n_items = len(interests)
            prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
            prob /= prob.sum()
            return np.random.choice(n_items, p=prob)

        # common item values (resale price)
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)

        # item compatibilities
        compats = np.triu(np.random.rand(self.n_items, self.n_items), k=1)
        compats = compats + compats.transpose()
        compats = compats / compats.sum(1)

        bids = []

        # create bids, one bidder at a time
        while len(bids) < self.n_bids:

            # bidder item values (buy price) and interests
            private_interests = np.random.rand(self.n_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)

            # substitutable bids of this bidder
            bidder_bids = {}

            # generate initial bundle, choose first item according to bidder interests
            prob = private_interests / private_interests.sum()
            item = np.random.choice(self.n_items, p=prob)
            bundle_mask = np.full(self.n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while np.random.rand() < self.add_item_prob:
                # stop when bundle full (no item left)
                if bundle_mask.sum() == self.n_items:
                    break
                item = choose_next_item(bundle_mask, private_interests, compats)
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            price = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

            # drop negatively priced bundles
            if price < 0:
                continue

            # bid on initial bundle
            bidder_bids[frozenset(bundle)] = price

            for bundle, price in bidder_bids.items():
                bids.append((list(bundle), price))

        bids_per_item = [[] for item in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        return {
            "bids": bids,
            "bids_per_item": bids_per_item
        }

    def get_bundle(self, compats, private_interests, private_values):
        item = np.random.choice(self.n_items, p=private_interests / private_interests.sum())
        bundle_mask = np.zeros(self.n_items, dtype=bool)
        bundle_mask[item] = True

        while np.random.rand() < self.add_item_prob and bundle_mask.sum() < self.n_items:
            compatibilities = compats[item] * private_interests
            compatibilities[bundle_mask] = 0
            next_item = np.random.choice(self.n_items, p=compatibilities / compatibilities.sum())
            bundle_mask[next_item] = True

        bundle = np.nonzero(bundle_mask)[0]
        price = private_values[bundle].sum() + len(bundle) ** (1 + self.additivity)
        return bundle, price

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']

        model = Model("CombinatorialAuction")

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
        'n_items': 500,  # Reduced number of items
        'n_bids': 1500,  # Reduced number of bids
        'min_value': 1,
        'max_value': 1000,  # Lowered maximum value for computational feasibility
        'value_deviation': 0.17,
        'additivity': 0.45,
        'add_item_prob': 0.73,
    }

    auction = CombinatorialAuction(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")