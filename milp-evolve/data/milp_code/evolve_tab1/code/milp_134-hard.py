import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ManufacturingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.min_product_value >= 0 and self.max_product_value >= self.min_product_value
        assert self.add_product_prob >= 0 and self.add_product_prob <= 1

        def choose_next_product(bundle_mask, interests, compats):
            n_products = len(interests)
            prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
            prob /= prob.sum()
            return np.random.choice(n_products, p=prob)

        values = self.min_product_value + (self.max_product_value - self.min_product_value) * np.random.rand(self.n_products)
        compats = np.triu(np.random.rand(self.n_products, self.n_products), k=1)
        compats = compats + compats.transpose()
        compats = compats / compats.sum(1)

        products = []
        n_extra_items = 0

        while len(products) < self.n_products:
            interest_level = np.random.rand(self.n_products)
            private_values = values + self.max_product_value * self.value_deviation * (2 * interest_level - 1)

            product_bids = {}

            prob = interest_level / interest_level.sum()
            item = np.random.choice(self.n_products, p=prob)
            bundle_mask = np.full(self.n_products, 0)
            bundle_mask[item] = 1

            while np.random.rand() < self.add_product_prob:
                if bundle_mask.sum() == self.n_products:
                    break
                item = choose_next_product(bundle_mask, interest_level, compats)
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]
            total_value = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

            if total_value < 0:
                continue

            product_bids[frozenset(bundle)] = total_value

            sub_candidates = []
            for item in bundle:
                bundle_mask = np.full(self.n_products, 0)
                bundle_mask[item] = 1

                while bundle_mask.sum() < len(bundle):
                    item = choose_next_product(bundle_mask, interest_level, compats)
                    bundle_mask[item] = 1

                sub_bundle = np.nonzero(bundle_mask)[0]
                sub_value = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + self.additivity)
                sub_candidates.append((sub_bundle, sub_value))

            budget = self.budget_factor * total_value
            min_resale_value = self.resale_factor * values[bundle].sum()
            for bundle, value in [
                sub_candidates[i] for i in np.argsort([-value for bundle, value in sub_candidates])]:

                if len(product_bids) >= self.max_sub_bids + 1 or len(products) + len(product_bids) >= self.n_products:
                    break

                if value < 0 or value > budget:
                    continue

                if values[bundle].sum() < min_resale_value:
                    continue

                if frozenset(bundle) in product_bids:
                    continue

                product_bids[frozenset(bundle)] = value

            if len(product_bids) > 2:
                extra_item = [self.n_products + n_extra_items]
                n_extra_items += 1
            else:
                extra_item = []

            for bundle, value in product_bids.items():
                products.append((list(bundle) + extra_item, value))

        products_per_item = [[] for item in range(self.n_products + n_extra_items)]
        for i, product in enumerate(products):
            bundle, value = product
            for item in bundle:
                products_per_item[item].append(i)

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            product1 = random.randint(0, len(products) - 1)
            product2 = random.randint(0, len(products) - 1)
            if product1 != product2:
                mutual_exclusivity_pairs.append((product1, product2))

        # Facility data generation
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=len(products)).tolist()
        capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()
        throughput = np.random.uniform(1.0, 5.0, size=len(products)).tolist()

        return {
            "products": products,
            "products_per_item": products_per_item,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "throughput": throughput
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        products = instance['products']
        products_per_item = instance['products_per_item']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        throughput = instance['throughput']

        model = Model("ManufacturingOptimization")

        manufacturing_vars = {i: model.addVar(vtype="B", name=f"Product_{i}") for i in range(len(products))}
        open_facility = {j: model.addVar(vtype="B", name=f"OpenFacility_{j}") for j in range(n_facilities)}
        product_to_facility = {(i, j): model.addVar(vtype="B", name=f"ProductFacility_{i}_{j}") for i in range(len(products)) for j in range(n_facilities)}
        facility_throughput = {i: model.addVar(vtype="C", name=f"Throughput_{i}") for i in range(len(products))}

        objective_expr = quicksum(value * manufacturing_vars[i] for i, (bundle, value) in enumerate(products)) \
                         - quicksum(operating_cost[j] * open_facility[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * quicksum(product_to_facility[i, j] for j in range(n_facilities)) for i in range(len(products))) \
                         - quicksum(setup_cost[j] * open_facility[j] for j in range(n_facilities))

        # Constraints: Each item can only be part of one accepted product
        for item, product_indices in enumerate(products_per_item):
            model.addCons(quicksum(manufacturing_vars[product_idx] for product_idx in product_indices) <= 1, f"Item_{item}")

        # Mutually exclusive product pairs
        for (product1, product2) in mutual_exclusivity_pairs:
            model.addCons(manufacturing_vars[product1] + manufacturing_vars[product2] <= 1, f"Exclusive_{product1}_{product2}")

        # Product assignment to facility
        for i in range(len(products)):
            model.addCons(quicksum(product_to_facility[i, j] for j in range(n_facilities)) == manufacturing_vars[i], f"ProductFacility_{i}")

        # Facility capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(product_to_facility[i, j] for i in range(len(products))) <= capacity[j] * open_facility[j], f"FacilityCapacity_{j}")

        # Throughput constraints
        for i in range(len(products)):
            model.addCons(facility_throughput[i] == quicksum(throughput[i] * product_to_facility[i, j] for j in range(n_facilities)), f"Throughput_{i}")

        ### new additional constraints
        max_throughput = np.max(throughput) * len(products)
        for j in range(n_facilities):
            model.addCons(quicksum(facility_throughput[i] * product_to_facility[i, j] for i in range(len(products))) <= max_throughput * open_facility[j], f"MaxThroughput_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_products': 100,
        'min_product_value': 3,
        'max_product_value': 5000,
        'value_deviation': 0.8,
        'additivity': 0.38,
        'add_product_prob': 0.52,
        'budget_factor': 1350.0,
        'resale_factor': 0.73,
        'max_sub_bids': 540,
        'n_exclusive_pairs': 2400,
        'facility_min_count': 1,
        'facility_max_count': 525,
    }

    optimization = ManufacturingOptimization(parameters, seed=42)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")