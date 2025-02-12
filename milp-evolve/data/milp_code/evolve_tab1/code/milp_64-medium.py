import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimpleCombinatorialAuction:
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

        # Common item values (resale price)
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)

        # Item compatibilities
        compats = np.triu(np.random.rand(self.n_items, self.n_items), k=1)
        compats = compats + compats.transpose()
        compats = compats / compats.sum(1)

        bids = []
        n_dummy_items = 0

        # Create bids, one bidder at a time
        while len(bids) < self.n_bids:
            # Bidder item values (buy price) and interests
            private_interests = np.random.rand(self.n_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)

            # Substitutable bids of this bidder
            bidder_bids = {}

            # Generate initial bundle, choose first item according to bidder interests
            prob = private_interests / private_interests.sum()
            item = np.random.choice(self.n_items, p=prob)
            bundle_mask = np.full(self.n_items, 0)
            bundle_mask[item] = 1

            # Add additional items, according to bidder interests and item compatibilities
            while np.random.rand() < self.add_item_prob:
                # Stop when bundle full (no item left)
                if bundle_mask.sum() == self.n_items:
                    break
                item = choose_next_item(bundle_mask, private_interests, compats)
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]

            # Compute bundle price with value additivity
            price = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

            # Drop negatively priced bundles
            if price < 0:
                continue

            # Bid on initial bundle
            bidder_bids[frozenset(bundle)] = price

            # Generate candidate substitutable bundles
            sub_candidates = []
            for item in bundle:
                # At least one item must be shared with initial bundle
                bundle_mask = np.full(self.n_items, 0)
                bundle_mask[item] = 1

                # Add additional items, according to bidder interests and item compatibilities
                while bundle_mask.sum() < len(bundle):
                    item = choose_next_item(bundle_mask, private_interests, compats)
                    bundle_mask[item] = 1

                sub_bundle = np.nonzero(bundle_mask)[0]

                # Compute bundle price with value additivity
                sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + self.additivity)

                sub_candidates.append((sub_bundle, sub_price))

            # Filter valid candidates, higher priced candidates first
            budget = self.budget_factor * price
            min_resale_value = self.resale_factor * values[bundle].sum()
            for bundle, price in [
                sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

                if len(bidder_bids) >= self.max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= self.n_bids:
                    break

                if price < 0 or price > budget:
                    # Negatively priced substitutable bundle avoided, over priced substitutable bundle avoided
                    continue

                if values[bundle].sum() < min_resale_value:
                    # Substitutable bundle below min resale value avoided
                    continue

                if frozenset(bundle) in bidder_bids:
                    # Duplicated substitutable bundle avoided
                    continue

                bidder_bids[frozenset(bundle)] = price

            # Add XOR constraint if needed (dummy item)
            if len(bidder_bids) > 2:
                dummy_item = [self.n_items + n_dummy_items]
                n_dummy_items += 1
            else:
                dummy_item = []

            for bundle, price in bidder_bids.items():
                bids.append((list(bundle) + dummy_item, price))

        bids_per_item = [[] for item in range(self.n_items + n_dummy_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        # New instance data for mutual exclusivity pairs
        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            bid1 = random.randint(0, len(bids) - 1)
            bid2 = random.randint(0, len(bids) - 1)
            if bid1 != bid2:
                mutual_exclusivity_pairs.append((bid1, bid2))

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']

        model = Model("SimpleCombinatorialAuction")

        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}

        # Objective: maximize the total price
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))

        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        # Indicator constraints for mutual exclusivity
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 2000,
        'n_bids': 600,
        'min_value': 40,
        'max_value': 2000,
        'value_deviation': 0.71,
        'additivity': 0.31,
        'add_item_prob': 0.59,
        'budget_factor': 13.0,
        'resale_factor': 0.75,
        'max_n_sub_bids': 1400,
        'n_exclusive_pairs': 1000,
    }

    auction = SimpleCombinatorialAuction(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")