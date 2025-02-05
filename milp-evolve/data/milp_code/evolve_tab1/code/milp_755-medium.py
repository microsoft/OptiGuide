import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedCombinatorialAuction:
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
        assert self.max_time >= self.min_time

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
        compats /= compats.sum(1)

        bids = []
        n_dummy_items = 0

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
            price = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

            if price < 0:
                continue
            bidder_bids[frozenset(bundle)] = price

            # generate candidates substitutable bundles
            sub_candidates = []
            for item in bundle:
                bundle_mask = np.full(self.n_items, 0)
                bundle_mask[item] = 1

                while bundle_mask.sum() < len(bundle):
                    item = choose_next_item(bundle_mask, private_interests, compats)
                    bundle_mask[item] = 1

                sub_bundle = np.nonzero(bundle_mask)[0]
                sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + self.additivity)
                sub_candidates.append((sub_bundle, sub_price))

            # filter valid candidates, higher priced candidates first
            budget = self.budget_factor * price
            min_resale_value = self.resale_factor * values[bundle].sum()
            for bundle, price in [
                sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

                if len(bidder_bids) >= self.max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= self.n_bids:
                    break

                if price < 0 or price > budget:
                    continue
                if values[bundle].sum() < min_resale_value:
                    continue

                if frozenset(bundle) in bidder_bids:
                    continue

                bidder_bids[frozenset(bundle)] = price

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

        bid_time_windows = [(
            self.min_time + (self.max_time - self.min_time) * np.random.rand(),
            self.min_time + (self.max_time - self.min_time) * np.random.rand()
        ) for _ in range(len(bids))]

        # Ensuring that start time is always less than end time
        bid_time_windows = [(min(tw[0], tw[1]), max(tw[0], tw[1])) for tw in bid_time_windows]

        # Item resources and bidder requirements
        resources = self.min_resource + (self.max_resource - self.min_resource) * np.random.rand(self.n_items + n_dummy_items)
        requirements = np.random.randint(self.min_requirement, self.max_requirement, size=(len(bids), self.n_resources))

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "bid_time_windows": bid_time_windows,
            "resources": resources,
            "requirements": requirements
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        bid_time_windows = instance['bid_time_windows']
        resources = instance['resources']
        requirements = instance['requirements']
        
        model = Model("EnhancedCombinatorialAuction")
        
        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        
        # Objective: maximize the total price and include penalty for overlapping bids
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))
        
        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        # Add time window constraints
        for i in range(len(bids)):
            start_time_i, end_time_i = bid_time_windows[i]
            for j in range(i+1, len(bids)):
                start_time_j, end_time_j = bid_time_windows[j]
                if not (end_time_i <= start_time_j or end_time_j <= start_time_i):
                    model.addCons(bid_vars[i] + bid_vars[j] <= 1, f"TimeConflict_{i}_{j}")

        # Add resource constraints
        for r in range(self.n_resources):
            model.addCons(quicksum(requirements[i][r] * bid_vars[i] for i in range(len(bids))) <= resources[r], f"Resource_{r}")

        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 500,
        'n_bids': 375,
        'min_value': 0,
        'max_value': 500,
        'value_deviation': 0.1,
        'additivity': 0.24,
        'add_item_prob': 0.59,
        'budget_factor': 13.5,
        'resale_factor': 0.59,
        'max_n_sub_bids': 3,
        'min_time': 0,
        'max_time': 50,
        'min_resource': 150,
        'max_resource': 700,
        'n_resources': 2,
        'min_requirement': 2,
        'max_requirement': 30,
    }

    auction = EnhancedCombinatorialAuction(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")