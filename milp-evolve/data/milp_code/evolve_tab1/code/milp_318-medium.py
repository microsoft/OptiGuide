import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class ClinicalTrialResourceAllocation:
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

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, n_facilities - 1)
            fac2 = random.randint(0, n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))

        facility_graph = nx.barabasi_albert_graph(n_facilities, 3)
        graph_edges = list(facility_graph.edges)

        emergency_repair_costs = np.random.uniform(50, 200, size=n_facilities).tolist()
        maintenance_periods = np.random.randint(1, 5, size=n_facilities).tolist()
        energy_consumption = np.random.normal(loc=100, scale=20, size=n_facilities).tolist()
        subcontracting_costs = np.random.uniform(300, 1000, size=self.n_bids).tolist()

        travel_distance_matrix = np.random.uniform(10, 1000, size=(n_facilities, n_facilities))
        demographic_data = np.random.dirichlet(np.ones(5), size=n_facilities).tolist()  # 5 demographic groups
        regional_supply_prices = np.random.normal(loc=100, scale=20, size=n_facilities).tolist()

        # New data for regional facility demands
        regional_facility_demand = np.random.uniform(50, 500, size=n_facilities).tolist()

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "minimum_bids": minimum_bids,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "graph_edges": graph_edges,
            "emergency_repair_costs": emergency_repair_costs,
            "maintenance_periods": maintenance_periods,
            "energy_consumption": energy_consumption,
            "subcontracting_costs": subcontracting_costs,
            "travel_distance_matrix": travel_distance_matrix,
            "demographic_data": demographic_data,
            "regional_supply_prices": regional_supply_prices,
            "regional_facility_demand": regional_facility_demand
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
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        graph_edges = instance['graph_edges']
        emergency_repair_costs = instance['emergency_repair_costs']
        maintenance_periods = instance['maintenance_periods']
        energy_consumption = instance['energy_consumption']
        subcontracting_costs = instance['subcontracting_costs']
        travel_distance_matrix = instance['travel_distance_matrix']
        demographic_data = instance['demographic_data']
        regional_supply_prices = instance['regional_supply_prices']
        regional_facility_demand = instance['regional_facility_demand']

        model = Model("ClinicalTrialResourceAllocation")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(bids)) for j in range(n_facilities)}
        subcontract_vars = {i: model.addVar(vtype="B", name=f"Subcontract_{i}") for i in range(len(bids))}
        
        travel_cost_vars = {j: model.addVar(vtype="C", name=f"TravelCost_{j}") for j in range(n_facilities)}
        demographic_vars = {(g, j): model.addVar(vtype="B", name=f"Demographic_{g}_{j}") for g in range(5) for j in range(n_facilities)}
        supply_cost_vars = {j: model.addVar(vtype="C", name=f"SupplyCost_{j}") for j in range(n_facilities)}
        regional_demand_vars = {j: model.addVar(vtype="C", name=f"RegionalDemand_{j}") for j in range(n_facilities)}

        # Additional variables for piecewise linear function
        capacity_usage_vars = {j: model.addVar(vtype="C", name=f"CapacityUsage_{j}") for j in range(n_facilities)}
        piecewise_vars = {(j, k): model.addVar(vtype="C", name=f"Piecewise_{j}_{k}", lb=0) for j in range(n_facilities) for k in range(self.num_segments)}

        # Constants for piecewise linear segments
        segment_points = np.linspace(0, 1, self.num_segments + 1)
        segment_slopes = [seg[1] - seg[0] for seg in zip(segment_points[:-1], segment_points[1:])]

        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price, complexity) in enumerate(bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * x_vars[i, j] for i in range(len(bids)) for j in range(n_facilities)) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(emergency_repair_costs[j] * y_vars[j] for j in range(n_facilities) if maintenance_periods[j] > 2) \
                         + quicksum(10 * bid_vars[i] for i in range(len(bids)) if minimum_bids[i % n_facilities] > 3) \
                         - quicksum(subcontracting_costs[i] * subcontract_vars[i] for i in range(len(bids))) \
                         - quicksum(travel_cost_vars[j] for j in range(n_facilities)) \
                         - quicksum(supply_cost_vars[j] for j in range(n_facilities)) \
                         - quicksum(regional_demand_vars[j] for j in range(n_facilities)) \
                         - quicksum(piecewise_vars[j, k] * segment_slopes[k] * operating_cost[j] for j in range(n_facilities) for k in range(self.num_segments))

        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        for i in range(len(bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i] - subcontract_vars[i], f"BidFacility_{i}")

        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(bids))) <= capacity[j] * y_vars[j], f"FacilityCapacity_{j}")

        bid_complexity_expr = quicksum(bids[i][2] * bid_vars[i] for i in range(len(bids)))
        max_complexity = quicksum(bids[i][2] for i in range(len(bids))) / 2
        model.addCons(bid_complexity_expr <= max_complexity, "MaxComplexity")

        for fac1, fac2 in mutual_exclusivity_pairs:
            model.addCons(y_vars[fac1] + y_vars[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        for (fac1, fac2) in graph_edges:
            model.addCons(y_vars[fac1] + y_vars[fac2] <= 1, f"FacilityGraph_{fac1}_{fac2}")

        for j in range(n_facilities):
            minimum_bids_sum = quicksum(bid_vars[i] for i in range(len(bids)) if x_vars[i, j])
            model.addCons(minimum_bids_sum >= minimum_bids[j] * y_vars[j], f"MinimumBids_{j}")

        for j in range(n_facilities):
            if maintenance_periods[j] > 0:
                model.addCons(y_vars[j] == 0, f"Maintenance_{j}")

        energy_expr = quicksum(energy_consumption[j] * y_vars[j] for j in range(n_facilities))
        max_energy = quicksum(energy_consumption[j] for j in range(n_facilities)) / 2
        model.addCons(energy_expr <= max_energy, "MaxEnergy")
        
        # New constraints for travel cost, demographic distribution, and supply cost
        for j in range(n_facilities):
            model.addCons(travel_cost_vars[j] == quicksum(travel_distance_matrix[j, k] * y_vars[k] for k in range(n_facilities)), f"TravelCost_{j}")

        for g in range(5):
            for j in range(n_facilities):
                model.addCons(demographic_vars[g, j] <= demographic_data[j][g] * y_vars[j], f"Demographic_{g}_{j}")

        for j in range(n_facilities):
            model.addCons(supply_cost_vars[j] == regional_supply_prices[j] * y_vars[j], f"SupplyCost_{j}")

        # New constraints for regional demand
        for j in range(n_facilities):
            model.addCons(regional_demand_vars[j] == regional_facility_demand[j] * quicksum(demographic_vars[g, j] for g in range(5)), f"RegionalDemand_{j}")

        # Piecewise linear function constraints
        for j in range(n_facilities):
            model.addCons(capacity_usage_vars[j] == quicksum(x_vars[i, j] for i in range(len(bids))) / capacity[j], f"CapacityUsage_{j}")
            for k in range(self.num_segments):
                model.addCons(piecewise_vars[j, k] >= capacity_usage_vars[j] - segment_points[k], f"Piecewise_{j}_{k}_ub")
                model.addCons(piecewise_vars[j, k] <= segment_points[k + 1] - segment_points[k], f"Piecewise_{j}_{k}_lb")
                model.addCons(piecewise_vars[j, k] <= capacity_usage_vars[j], f"Piecewise_{j}_{k}_cons")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 1000,
        'n_bids': 1125,
        'min_value': 2250,
        'max_value': 3000,
        'max_bundle_size': 400,
        'add_item_prob': 0.24,
        'facility_min_count': 315,
        'facility_max_count': 1890,
        'n_exclusive_pairs': 295,
        'num_segments': 100,
    }
    auction = ClinicalTrialResourceAllocation(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")