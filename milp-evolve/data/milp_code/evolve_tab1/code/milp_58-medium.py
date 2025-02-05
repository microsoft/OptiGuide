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
        n_dummy_items = 0

        while len(bids) < self.n_bids:
            private_interests = np.random.rand(self.n_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)
            bidder_bids = {}
            prob = private_interests / private_interests.sum()
            item = np.random.choice(self.n_items, p=prob)
            bundle_mask = np.full(self.n_items, 0)
            bundle_mask[item] = 1

            while np.random.rand() < self.add_item_prob:
                if bundle_mask.sum() == self.n_items:
                    break
                item = choose_next_item(bundle_mask, private_interests, compats)
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]
            base_price = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

            for scenario in range(self.n_scenarios):
                price = base_price * (1 + np.random.normal(self.price_mean, self.price_std))
                if price < 0:
                    continue
                bidder_bids[frozenset(bundle)] = price

            sub_candidates = []
            for item in bundle:
                bundle_mask = np.full(self.n_items, 0)
                bundle_mask[item] = 1
                while bundle_mask.sum() < len(bundle):
                    item = choose_next_item(bundle_mask, private_interests, compats)
                    bundle_mask[item] = 1

                sub_bundle = np.nonzero(bundle_mask)[0]
                base_sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + self.additivity)
                sub_candidates.append((sub_bundle, base_sub_price))

            budget = self.budget_factor * base_price
            min_resale_value = self.resale_factor * values[bundle].sum()
            for bundle, base_sub_price in [
                sub_candidates[i] for i in np.argsort([-base_sub_price for bundle, base_sub_price in sub_candidates])]:

                if len(bidder_bids) >= self.max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= self.n_bids:
                    break

                for scenario in range(self.n_scenarios):
                    sub_price = base_sub_price * (1 + np.random.normal(self.price_mean, self.price_std))
                    if sub_price < 0 or sub_price > budget:
                        continue
                    if values[bundle].sum() < min_resale_value:
                        continue
                    if frozenset(bundle) in bidder_bids:
                        continue
                    bidder_bids[frozenset(bundle)] = sub_price

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
        
        model = Model("CombinatorialAuction")
        
        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        waste_vars = {i: model.addVar(vtype="C", name=f"Waste_{i}") for i in range(len(bids))}

        # Objective: maximize the robust total price and minimize waste
        total_waste = quicksum(waste_vars[i] for i in range(len(bids)))
        total_value_robust = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))
        objective_expr = total_value_robust - self.waste_penalty * total_waste
        
        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")
        
        # Indicator constraints for mutual exclusivity
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")
        
        # Add constraints linking waste production with bids
        for i in range(len(bids)):
            bundle, price = bids[i]
            model.addCons(waste_vars[i] >= 0, f"Waste_LB_{i}")
            model.addCons(waste_vars[i] >= self.waste_factor * (1 - bid_vars[i]), f"Waste_Link_{i}")

        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 2000,
        'n_bids': 2500,
        'min_value': 72,
        'max_value': 1800,
        'value_deviation': 0.24,
        'additivity': 0.8,
        'add_item_prob': 0.38,
        'budget_factor': 300.0,
        'resale_factor': 0.59,
        'max_n_sub_bids': 2450,
        'n_exclusive_pairs': 375,
        'waste_penalty': 0.73,
        'waste_factor': 0.1,
        'price_mean': 0.0,
        'price_std': 0.66,
        'n_scenarios': 50,
    }

    auction = CombinatorialAuction(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")