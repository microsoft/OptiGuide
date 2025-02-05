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

            # add XOR constraint if needed (dummy item)
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

        # Create additional data for resource and proficiency complexities
        resource_usage = {i: np.random.uniform(self.min_resource_usage, self.max_resource_usage) for i in range(len(bids))}
        proficiency_groups = [set(random.sample(range(len(bids)), len(bids) // self.num_proficiency_groups)) for _ in range(self.num_proficiency_groups)]
        employee_efficiencies = {i: np.random.normal(self.avg_efficiency, self.std_efficiency) for i in range(len(bids))}

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "resource_usage": resource_usage,
            "proficiency_groups": proficiency_groups,
            "employee_efficiencies": employee_efficiencies
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        resource_usage = instance['resource_usage']
        proficiency_groups = instance['proficiency_groups']
        employee_efficiencies = instance['employee_efficiencies']
        
        model = Model("CombinatorialAuction")

        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        
        # New decision variables for the resource and proficiency complexities
        resource_vars = {i: model.addVar(vtype="C", name=f"Resource_{i}") for i in range(len(bids))}
        group_vars = {(i, g): model.addVar(vtype="B", name=f"Group_{g}_{i}") for i in range(len(bids)) for g in range(self.num_proficiency_groups)}
        
        # Objective: maximize the total price
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))

        # Adjusted objective to include employee efficiencies and resource usage
        objective_expr += quicksum(employee_efficiencies[i] * bid_vars[i] for i in range(len(bids)))
        objective_expr -= quicksum(resource_usage[i] * resource_vars[i] for i in range(len(bids)))
        
        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")
        
        # Ensure selected bids do not exceed resource limits
        model.addCons(quicksum(resource_vars[i] * bid_vars[i] for i in range(len(bids))) <= self.resource_limit)
        
        # Ensure proficiency groups influence bid selection
        for g in range(self.num_proficiency_groups):
            model.addCons(quicksum(group_vars[(i, g)] * bid_vars[i] for i in range(len(bids))) <= len(proficiency_groups[g]))
        
        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 700,
        'n_bids': 500,
        'min_value': 5,
        'max_value': 900,
        'value_deviation': 0.31,
        'additivity': 0.59,
        'add_item_prob': 0.66,
        'budget_factor': 1.12,
        'resale_factor': 0.73,
        'max_n_sub_bids': 45,
        'min_resource_usage': 5,
        'max_resource_usage': 7,
        'num_proficiency_groups': 7,
        'avg_efficiency': 5,
        'std_efficiency': 0,
        'resource_limit': 900,
    }

    auction = CombinatorialAuction(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")