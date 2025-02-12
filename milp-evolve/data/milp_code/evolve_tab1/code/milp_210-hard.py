import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class AdvancedCombinatorialAuctionWithFLPMutualExclusivity:
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

        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)
        bids = []

        while len(bids) < self.n_bids:
            bundle_size = np.random.randint(1, self.max_bundle_size + 1)
            bundle = np.random.choice(self.n_items, size=bundle_size, replace=False)
            price = values[bundle].sum()
            complexity = np.random.poisson(lam=5)

            if price < 0:
                continue

            bids.append((bundle.tolist(), price, complexity))

        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price, complexity = bid
            for item in bundle:
                bids_per_item[item].append(i)

        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=len(bids)).tolist()
        capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()
        maintenance_cost = np.random.lognormal(mean=3, sigma=1.0, size=n_facilities).tolist()

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, n_facilities - 1)
            fac2 = random.randint(0, n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))

        set_packing_constraints = []
        if self.knapsack_constraints > 0:
            for _ in range(self.knapsack_constraints):
                knapsack_bids = np.random.choice(len(bids), size=np.random.randint(2, 5), replace=False).tolist()
                max_items = np.random.randint(1, len(knapsack_bids))
                set_packing_constraints.append((knapsack_bids, max_items))

        # New - Generate worker and shift data
        skill_levels = np.random.gamma(shape=2.0, scale=1.0, size=self.n_workers).tolist()
        shift_prefs = {worker: np.random.choice(3, p=[0.6, 0.3, 0.1]) for worker in range(self.n_workers)}  # Probability of preferring shift 0, 1, 2

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "maintenance_cost": maintenance_cost,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "set_packing_constraints": set_packing_constraints,
            "skill_levels": skill_levels,
            "shift_prefs": shift_prefs
        }

    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        maintenance_cost = instance['maintenance_cost']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        set_packing_constraints = instance['set_packing_constraints']
        skill_levels = instance['skill_levels']
        shift_prefs = instance['shift_prefs']

        model = Model("AdvancedCombinatorialAuctionWithFLPMutualExclusivity")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(bids)) for j in range(n_facilities)}
        facility_workload = {j: model.addVar(vtype="I", name=f"workload_{j}", lb=0) for j in range(n_facilities)}

        # New - Shift allocation variables
        shift_vars = {(j, s): model.addVar(vtype="I", name=f"shift_{j}_{s}", lb=0) for j in range(n_facilities) for s in range(self.n_shifts)}

        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price, complexity) in enumerate(bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * x_vars[i, j] for i in range(len(bids)) for j in range(n_facilities)) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(maintenance_cost[j] * facility_workload[j] for j in range(n_facilities)) \
                         - quicksum(complexity * bid_vars[i] for i, (bundle, price, complexity) in enumerate(bids)) \
                         - quicksum(self.overtime_cost_per_hour * shift_vars[j, s] for j in range(n_facilities) for s in range(self.n_shifts) if s > self.regular_shift_hours)

        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        for i in range(len(bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i], f"BidFacility_{i}")

        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(bids))) <= capacity[j] * y_vars[j], f"FacilityCapacity_{j}")

        for j in range(n_facilities):
            model.addCons(facility_workload[j] == quicksum(x_vars[i, j] * bids[i][2] for i in range(len(bids))), f"Workload_{j}")

        max_complexity = quicksum(bids[i][2] for i in range(len(bids))) / 2
        model.addCons(quicksum(bids[i][2] * bid_vars[i] for i in range(len(bids))) <= max_complexity, "MaxComplexity")

        for fac1, fac2 in mutual_exclusivity_pairs:
            model.addCons(y_vars[fac1] + y_vars[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        for constraint_num, (knapsack_bids, max_items) in enumerate(set_packing_constraints):
            model.addCons(quicksum(bid_vars[bid] for bid in knapsack_bids) <= max_items, f"SetPacking_{constraint_num}")

        # New - Maximum shift hours constraint
        for j in range(n_facilities):
            for s in range(self.n_shifts):
                model.addCons(shift_vars[j, s] <= self.max_shift_hours, f"MaxShiftHours_{j}_{s}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 1875,
        'n_bids': 37,
        'min_value': 1350,
        'max_value': 5000,
        'max_bundle_size': 175,
        'add_item_prob': 0.31,
        'facility_min_count': 150,
        'facility_max_count': 1875,
        'complexity_mean': 75,
        'complexity_stddev': 150,
        'n_exclusive_pairs': 111,
        'knapsack_constraints': 150,
        'n_workers': 700,
        'n_shifts': 45,
        'max_shift_hours': 72,
        'overtime_cost_per_hour': 11,
        'regular_shift_hours': 6,
    }

    auction = AdvancedCombinatorialAuctionWithFLPMutualExclusivity(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")