import random
import time
import numpy as np
from itertools import product
from pyscipopt import Model, quicksum

class CombinedAuctionLogisticsOptimization:
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

        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)
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
        transportation_capacity = np.random.poisson(10, size=self.n_items)
        activation_costs = np.random.normal(100, 20, size=len(bids))
        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)
        
        FacilityCost = np.random.randint(1000, 5000, self.n_facilities)
        FacilityOperatingCost = np.random.randint(10, 50, self.n_facilities)
        TransportCost = np.random.randint(1, 10, (self.n_facilities, self.n_items))
        FacilityCapacity = np.random.randint(50, 100, self.n_facilities)

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "transportation_capacity": transportation_capacity,
            "activation_costs": activation_costs,
            "FacilityCost": FacilityCost,
            "FacilityOperatingCost": FacilityOperatingCost,
            "TransportCost": TransportCost,
            "FacilityCapacity": FacilityCapacity
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        transportation_capacity = instance['transportation_capacity']
        activation_costs = instance['activation_costs']
        FacilityCost = instance['FacilityCost']
        FacilityOperatingCost = instance['FacilityOperatingCost']
        TransportCost = instance['TransportCost']
        FacilityCapacity = instance['FacilityCapacity']

        model = Model("CombinedAuctionLogisticsOptimization")
        
        bid_vars = {i: model.addVar(vtype="C", lb=0, name=f"Bid_{i}") for i in range(len(bids))}
        activate_vars = {i: model.addVar(vtype="B", name=f"Activate_{i}") for i in range(len(bids))}
        facility_open = {f: model.addVar(vtype="B", name=f"FacilityOpen_{f}") for f in range(self.n_facilities)}
        assigned_bid = {(f, i): model.addVar(vtype="B", name=f"AssignedBid_{f}_{i}") for f in range(self.n_facilities) for i in range(self.n_items)}

        objective_expr = (
            quicksum(price * bid_vars[i] - activation_costs[i] * activate_vars[i] for i, (bundle, price) in enumerate(bids))
            - quicksum(FacilityCost[f] * facility_open[f] for f in range(self.n_facilities))
            - quicksum(FacilityOperatingCost[f] * assigned_bid[f, i] for f in range(self.n_facilities) for i in range(self.n_items))
            - quicksum(TransportCost[f, i] * assigned_bid[f, i] for f in range(self.n_facilities) for i in range(self.n_items))
        )

        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= transportation_capacity[item], f"Item_{item}")

        for i in range(len(bids)):
            model.addCons(bid_vars[i] <= 1000 * activate_vars[i], f"Activation_Bid_{i}")

        for f in range(self.n_facilities):
            model.addCons(quicksum(assigned_bid[f, i] for i in range(self.n_items)) <= FacilityCapacity[f] * facility_open[f], f"FacilityCapacity_{f}")

        for i in range(self.n_items):
            model.addCons(quicksum(assigned_bid[f, i] for f in range(self.n_facilities)) == 1, f"AssignItem_{i}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 750,
        'n_bids': 150,
        'min_value': 1,
        'max_value': 56,
        'value_deviation': 0.66,
        'additivity': 0.52,
        'add_item_prob': 0.52,
        'n_facilities': 25,
    }
    
    combined = CombinedAuctionLogisticsOptimization(parameters, seed=seed)
    instance = combined.generate_instance()
    solve_status, solve_time = combined.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")