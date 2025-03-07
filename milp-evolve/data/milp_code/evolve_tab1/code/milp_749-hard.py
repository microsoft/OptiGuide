import random
import time
import numpy as np
from scipy.stats import poisson, binom
from pyscipopt import Model, quicksum

class EnhancedCombinatorialAuction:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        assert self.min_value >= 0 and self.max_value >= self.min_value
        assert self.add_item_prob >= 0 and self.add_item_prob <= 1

        # Common item values (resale price)
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)

        # Bids generation
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
            price = private_values[bundle].sum() + len(bundle) ** (1 + self.additivity)

            if price > 0:
                bids.append((list(bundle), price))

        # Capacity constraints and activation costs
        transportation_capacity = poisson.rvs(mu=10, size=self.n_items)
        activation_costs = np.random.normal(loc=100, scale=20, size=len(bids))

        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "transportation_capacity": transportation_capacity,
            "activation_costs": activation_costs
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        transportation_capacity = instance['transportation_capacity']
        activation_costs = instance['activation_costs']
        
        model = Model("EnhancedCombinatorialAuction")
        
        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        activate_vars = {i: model.addVar(vtype="B", name=f"Activate_{i}") for i in range(len(bids))}
        
        # Objective: maximize the total price minus activation costs
        objective_expr = quicksum(price * bid_vars[i] - activation_costs[i] * activate_vars[i] for i, (bundle, price) in enumerate(bids))
        
        # Constraints: Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")
        
        # Capacity constraints for each item based on transportation scenario
        for i, capacity in enumerate(transportation_capacity):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bids_per_item[i]) <= capacity, f"Capacity_Item_{i}")
        
        # Ensure activation variable is set if any bids for the bundle is accepted
        for i in range(len(bids)):
            model.addCons(bid_vars[i] <= activate_vars[i], f"Activation_Bid_{i}")
        
        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 2000,
        'n_bids': 675,
        'min_value': 50,
        'max_value': 100,
        'value_deviation': 0.17,
        'additivity': 0.66,
        'add_item_prob': 0.8,
        'min_teams': 210,
        'max_teams': 3000,
        'min_locations': 11,
        'max_locations': 800,
    }

    auction = EnhancedCombinatorialAuction(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")