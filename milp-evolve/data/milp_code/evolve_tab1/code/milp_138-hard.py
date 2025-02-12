import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ComplexMILP:
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

        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)
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
            price = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

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
                sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + self.additivity)
                sub_candidates.append((sub_bundle, sub_price))

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

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            bid1 = random.randint(0, len(bids) - 1)
            bid2 = random.randint(0, len(bids) - 1)
            if bid1 != bid2:
                mutual_exclusivity_pairs.append((bid1, bid2))

        # Facility data generation
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=len(bids)).tolist()
        capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()
        throughput = np.random.uniform(1.0, 5.0, size=len(bids)).tolist()

        # Adding transportation costs between facilities and seasonal variation
        transportation_costs = np.random.randint(1, 20, size=(n_facilities, n_facilities)).tolist()
        seasonal_variation = np.random.normal(0, 0.1, size=n_facilities).tolist()

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "throughput": throughput,
            "transportation_costs": transportation_costs,
            "seasonal_variation": seasonal_variation
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        throughput = instance['throughput']
        transportation_costs = instance['transportation_costs']
        seasonal_variation = instance['seasonal_variation']

        model = Model("ComplexMILP")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(len(bids)) for j in range(n_facilities)}
        t_vars = {i: model.addVar(vtype="C", name=f"throughput_{i}") for i in range(len(bids))}
        flow_vars = {(j, k): model.addVar(vtype="C", name=f"flow_{j}_{k}") for j in range(n_facilities) for k in range(n_facilities)}
        aux_vars = {(j, k): model.addVar(vtype="C", name=f"aux_{j}_{k}") for j in range(n_facilities) for k in range(n_facilities)}

        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * quicksum(x_vars[i, j] for j in range(n_facilities)) for i in range(len(bids))) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(transportation_costs[j][k] * flow_vars[j, k] for j in range(n_facilities) for k in range(n_facilities))

        # Constraints: Each item can only be part of one accepted bid
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        # Mutually exclusive bid pairs
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")

        # Bid assignment to facility
        for i in range(len(bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i], f"BidFacility_{i}")

        # Facility capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(bids))) <= capacity[j] * y_vars[j], f"FacilityCapacity_{j}")

        # Throughput constraints
        for i in range(len(bids)):
            model.addCons(t_vars[i] == quicksum(throughput[i] * x_vars[i, j] for j in range(n_facilities)), f"Throughput_{i}")

        # New Constraints: Flow capacity and transportation constraints
        for j in range(n_facilities):
            slope = seasonal_variation[j]
            for k in range(n_facilities):
                model.addCons(flow_vars[j, k] <= aux_vars[j, k], f"supply_capacity_aux_{j}_{k}")
                model.addCons(aux_vars[j, k] <= capacity[j] * (1 + slope) * y_vars[j], f"supply_convex_{j}_{k}_1")
                model.addCons(aux_vars[j, k] >= 0, f"supply_convex_{j}_{k}_2")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 2500,
        'n_bids': 100,
        'min_value': 8,
        'max_value': 5000,
        'value_deviation': 0.73,
        'additivity': 0.8,
        'add_item_prob': 0.61,
        'budget_factor': 750.0,
        'resale_factor': 0.8,
        'max_n_sub_bids': 3000,
        'n_exclusive_pairs': 2400,
        'facility_min_count': 18,
        'facility_max_count': 75,
    }

    complex_milp = ComplexMILP(parameters, seed=seed)
    instance = complex_milp.generate_instance()
    solve_status, solve_time = complex_milp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")