import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ElectricVehicleChargingAllocation:
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
        minimum_bids = np.random.randint(1, 5, size=n_facilities).tolist()
        emergency_repair_costs = np.random.uniform(50, 200, size=n_facilities).tolist()
        maintenance_periods = np.random.randint(1, 5, size=n_facilities).tolist()
        subcontracting_costs = np.random.uniform(300, 1000, size=self.n_bids).tolist()
        renewable_energy = np.random.uniform(0, 1, size=n_facilities).tolist()
        charging_demand = np.random.uniform(1, 5, size=len(bids)).tolist()
        satisfaction_levels = np.random.uniform(0, 1, size=n_facilities).tolist()
        environmental_penalties = np.random.uniform(0, 50, size=n_facilities).tolist()
        peak_demand_periods = np.random.randint(2, 5)
        high_demand_rates = np.random.uniform(1.5, 3.0, size=peak_demand_periods).tolist()
        medium_demand_rates = np.random.uniform(1.0, 2.0, size=peak_demand_periods).tolist()
        low_demand_rates = np.random.uniform(0.5, 1.5, size=peak_demand_periods).tolist()

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "minimum_bids": minimum_bids,
            "emergency_repair_costs": emergency_repair_costs,
            "maintenance_periods": maintenance_periods,
            "subcontracting_costs": subcontracting_costs,
            "renewable_energy": renewable_energy,
            "charging_demand": charging_demand,
            "satisfaction_levels": satisfaction_levels,
            "environmental_penalties": environmental_penalties,
            "high_demand_rates": high_demand_rates,
            "medium_demand_rates": medium_demand_rates,
            "low_demand_rates": low_demand_rates,
        }

    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        minimum_bids = instance['minimum_bids']
        emergency_repair_costs = instance['emergency_repair_costs']
        maintenance_periods = instance['maintenance_periods']
        subcontracting_costs = instance['subcontracting_costs']
        renewable_energy = instance['renewable_energy']
        charging_demand = instance['charging_demand']
        satisfaction_levels = instance['satisfaction_levels']
        environmental_penalties = instance['environmental_penalties']
        high_demand_rates = instance['high_demand_rates']
        medium_demand_rates = instance['medium_demand_rates']
        low_demand_rates = instance['low_demand_rates']

        model = Model("ElectricVehicleChargingAllocation")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(bids)) for j in range(n_facilities)}
        subcontract_vars = {i: model.addVar(vtype="B", name=f"Subcontract_{i}") for i in range(len(bids))}
        upgrade_vars = {j: model.addVar(vtype="B", name=f"Upgrade_{j}") for j in range(n_facilities)}
        satisfaction_vars = {j: model.addVar(vtype="C", name=f"Satisfaction_{j}") for j in range(n_facilities)}

        # Piecewise linear charging rate models
        demand_periods = range(len(high_demand_rates))
        high_rate_vars = {(i, p): model.addVar(vtype="C", name=f"HighRate_{i}_{p}", lb=0) for i in range(len(bids)) for p in demand_periods}
        medium_rate_vars = {(i, p): model.addVar(vtype="C", name=f"MediumRate_{i}_{p}", lb=0) for i in range(len(bids)) for p in demand_periods}
        low_rate_vars = {(i, p): model.addVar(vtype="C", name=f"LowRate_{i}_{p}", lb=0) for i in range(len(bids)) for p in demand_periods}

        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price, complexity) in enumerate(bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * x_vars[i, j] for i in range(len(bids)) for j in range(n_facilities)) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(emergency_repair_costs[j] * y_vars[j] for j in range(n_facilities) if maintenance_periods[j] > 2) \
                         + 10 * quicksum(y_vars[j] for j in range(n_facilities) if minimum_bids[j] > 3) \
                         - quicksum(subcontracting_costs[i] * subcontract_vars[i] for i in range(len(bids))) \
                         - 5 * quicksum(upgrade_vars[j] for j in range(n_facilities)) \
                         - quicksum((charging_demand[i] * high_demand_rates[p] * high_rate_vars[i, p] +
                                     charging_demand[i] * medium_demand_rates[p] * medium_rate_vars[i, p] + 
                                     charging_demand[i] * low_demand_rates[p] * low_rate_vars[i, p]) 
                                     for i in range(len(bids)) for p in demand_periods) \
                         - quicksum(environmental_penalties[j] * (1 - renewable_energy[j]) for j in range(n_facilities))

        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        for i in range(len(bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i] - subcontract_vars[i], f"BidFacility_{i}")

        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(bids))) <= capacity[j] * y_vars[j], f"FacilityCapacity_{j}")

        for j in range(n_facilities):
            minimum_bids_sum = quicksum(bid_vars[i] for i in range(len(bids)) if x_vars[i, j])
            model.addCons(minimum_bids_sum >= minimum_bids[j] * y_vars[j], f"MinimumBids_{j}")

        model.addCons(quicksum(bid_vars[i] * bidding_complexity for i, (bundle, price, bidding_complexity) in enumerate(bids)) <= sum(bidding_complexity for _, _, bidding_complexity in bids) / 2, "MaxComplexity")

        for j in range(n_facilities):
            if maintenance_periods[j] > 0:
                model.addCons(y_vars[j] == 0, f"Maintenance_{j}")

        for j in range(n_facilities):
            model.addCons(upgrade_vars[j] <= y_vars[j], f"UpgradeOnlyIfOperational_{j}")

        # New constraints for piecewise linear formulations
        for i in range(len(bids)):
            for p in demand_periods:
                if p == 0:
                    model.addCons(high_rate_vars[i, p] + medium_rate_vars[i, p] + low_rate_vars[i, p] == x_vars[i, p], f"PiecewiseRate_{i}_{p}")
                else:
                    model.addCons(high_rate_vars[i, p] + medium_rate_vars[i, p] + low_rate_vars[i, p] == x_vars[i, p] - x_vars[i, p-1], f"PiecewiseRate_{i}_{p}")

        # Add constraints for the Big M formulation
        M = 1000  # Big M value
        for j in range(n_facilities):
            model.addCons(y_vars[j] <= upgrade_vars[j] + (1 - y_vars[j]) * M, f"BigM_Upgrade_{j}")
            model.addCons((1 - upgrade_vars[j]) * M >= y_vars[j], f"BigM_NotUpgrade_{j}")

        # Symmetry-breaking constraints
        for j in range(1, n_facilities):
            model.addCons(y_vars[j - 1] >= y_vars[j], f"Symmetry_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 1124,
        'n_bids': 1125,
        'min_value': 1050,
        'max_value': 3000,
        'max_bundle_size': 840,
        'add_item_prob': 0.45,
        'facility_min_count': 354,
        'facility_max_count': 531,
        'peak_demand_periods': 2,
    }
    
    n_facilities = random.randint(945, 1417)

    auction = ElectricVehicleChargingAllocation(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")