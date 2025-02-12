import random
import time
import numpy as np
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

            # drop negatively priced bundles
            if price < 0:
                continue

            # bid on initial bundle
            bidder_bids[frozenset(bundle)] = price

            for bundle, price in bidder_bids.items():
                bids.append((list(bundle), price))

        bids_per_item = [[] for item in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        station_capacities = np.random.randint(self.station_capacity_interval[0], self.station_capacity_interval[1], self.n_stations)
        
        # Generate labor availability and demand data
        labor_availability = np.random.randint(self.labor_availability_min, self.labor_availability_max, self.n_days)
        energy_consumption_per_product = np.random.rand(self.n_items)
        customer_demand = np.random.randint(self.customer_demand_min, self.customer_demand_max, self.n_items)

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "station_capacities": station_capacities,
            "labor_availability": labor_availability,
            "energy_consumption_per_product": energy_consumption_per_product,
            "customer_demand": customer_demand
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        station_capacities = instance['station_capacities']
        labor_availability = instance['labor_availability']
        energy_consumption_per_product = instance['energy_consumption_per_product']
        customer_demand = instance['customer_demand']

        model = Model("CombinatorialAuction")

        # Decision variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        open_stations = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(self.n_stations)}
        labor_hours = {d: model.addVar(vtype="I", name=f"LaborHours_{d}") for d in range(self.n_days)}
        production_qty = {i: model.addVar(vtype="I", name=f"ProductionQty_{i}") for i in range(self.n_items)}

        # Objective: maximize the total price while minimizing energy consumption
        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))
        energy_penalty = quicksum(energy_consumption_per_product[i] * production_qty[i] for i in range(self.n_items))
        model.setObjective(objective_expr - self.energy_cost * energy_penalty, "maximize")

        # Ensure each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        # Capacity constraints for each station
        for j in range(self.n_stations):
            model.addCons(quicksum(bid_vars[i] for i, (bundle, price) in enumerate(bids) if j in bundle) <= station_capacities[j] * open_stations[j], f"StationCapacity_{j}")

        # Add labor availability constraints
        for d in range(self.n_days):
            model.addCons(labor_hours[d] <= labor_availability[d], f"LaborAvailability_{d}")

        # Ensure production meets customer demand
        for i in range(self.n_items):
            model.addCons(production_qty[i] >= customer_demand[i], f"CustomerDemand_{i}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 1125,
        'n_bids': 1687,
        'min_value': 150,
        'max_value': 1686,
        'value_deviation': 0.52,
        'additivity': 0.45,
        'add_item_prob': 0.66,
        'n_stations': 350,
        'station_capacity_interval': (75, 3000),
        'labor_availability_min': 300,
        'labor_availability_max': 500,
        'n_days': 150,
        'energy_cost': 15.0,
        'customer_demand_min': 7,
        'customer_demand_max': 50,
    }

    auction = CombinatorialAuction(parameters, seed)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")