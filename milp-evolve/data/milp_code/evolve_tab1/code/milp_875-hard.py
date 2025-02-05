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

            # bid on initial bundle
            bidder_bids[frozenset(bundle)] = price

            for bundle, price in bidder_bids.items():
                bids.append((list(bundle), price))

        bids_per_item = [[] for item in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        # Generate scenarios for stochastic and robust optimization
        scenarios = []
        for s in range(self.n_scenarios):
            scenario_interests = np.random.rand(self.n_items)
            scenario_values = values + self.max_value * self.value_deviation * (2 * scenario_interests - 1)
            scenarios.append({
                "interests": scenario_interests,
                "values": scenario_values
            })

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "scenarios": scenarios
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        scenarios = instance['scenarios']

        model = Model("SimplifiedCombinatorialAuction")

        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        scenario_vars = {s: model.addVar(vtype="C", name=f"Scenario_{s}") for s in range(len(scenarios))}

        # Objective: maximize the expected total price across all scenarios
        objective_expr = quicksum(scenario_vars[s] for s in scenario_vars) / len(scenarios)

        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        # Robust constraints for each scenario
        for s, scenario in enumerate(scenarios):
            scenario_price = quicksum(
                scenario['values'][item] * bid_vars[i]
                for i, (bundle, price) in enumerate(bids)
                for item in bundle
            )
            model.addCons(scenario_vars[s] == scenario_price)
        
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 750,
        'n_bids': 1500,
        'min_value': 5,
        'max_value': 225,
        'value_deviation': 0.52,
        'additivity': 0.59,
        'add_item_prob': 0.66,
        'n_scenarios': 25,
    }

    auction = SimplifiedCombinatorialAuction(parameters, seed)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")