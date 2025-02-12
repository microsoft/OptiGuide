import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class AdvancedCombinatorialAuctionWithFLPBigM:
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

        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=len(bids)).tolist()
        capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()
        maintenance_cost = np.random.lognormal(mean=3, sigma=1.0, size=n_facilities).tolist()
        
        skill_levels = np.random.gamma(shape=2.0, scale=1.0, size=self.n_workers).tolist()
        shift_prefs = {worker: np.random.choice(3, p=[0.6, 0.3, 0.1]) for worker in range(self.n_workers)}

        res = {
            "bids": bids,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "maintenance_cost": maintenance_cost,
            "skill_levels": skill_levels,
            "shift_prefs": shift_prefs
        }
        return res

    def solve(self, instance):
        bids = instance['bids']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        maintenance_cost = instance['maintenance_cost']
        skill_levels = instance['skill_levels']
        shift_prefs = instance['shift_prefs']

        model = Model("AdvancedCombinatorialAuctionWithFLPBigM")
        
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(bids)) for j in range(n_facilities)}
        facility_workload = {j: model.addVar(vtype="I", name=f"workload_{j}", lb=0) for j in range(n_facilities)}
        shift_vars = {(j, s): model.addVar(vtype="I", name=f"shift_{j}_{s}", lb=0) for j in range(n_facilities) for s in range(self.n_shifts)}

        # New - Big M constant for logical constraints
        M = 1e6

        # Updated objective to be more nuanced and complex
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price, complexity) in enumerate(bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * x_vars[i, j] for i in range(len(bids)) for j in range(n_facilities)) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(maintenance_cost[j] * facility_workload[j] for j in range(n_facilities)) \
                         - quicksum(skill_levels[worker] * shift_vars[j, s] for j in range(n_facilities) for s in range(self.n_shifts) for worker in shift_prefs if shift_prefs[worker] == s) 

        # Orders consolidated constraints and logical constraints for added complexity
        for item in range(len(bids)):
            model.addCons(quicksum(bid_vars[bid_index] for bid_index in range(len(bids)) if item in bids[bid_index][0]) <= 1)
        
        for i in range(len(bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i])
        
        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(bids))) <= capacity[j] * y_vars[j])
        
        for j in range(n_facilities):
            model.addCons(facility_workload[j] == quicksum(x_vars[i, j] * bids[i][2] for i in range(len(bids))))

        for j in range(n_facilities):
            for s in range(self.n_shifts):
                model.addCons(shift_vars[j, s] <= self.max_shift_hours)

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 900,
        'n_bids': 250,
        'min_value': 35,
        'max_value': 40,
        'max_bundle_size': 200,
        'add_item_prob': 0.77,
        'facility_min_count': 10,
        'facility_max_count': 140,
        'complexity_mean': 21,
        'complexity_stddev': 4,
        'n_exclusive_pairs': 525,
        'knapsack_constraints': 100,
        'n_workers': 15,
        'n_shifts': 6,
        'max_shift_hours': 400,
        'overtime_cost_per_hour': 13.5,
        'regular_shift_hours': 7,
    }

    auction = AdvancedCombinatorialAuctionWithFLPBigM(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")