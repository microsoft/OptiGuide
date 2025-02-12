import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class AdvancedResourceAllocationWithComplexities:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data generation #################
    def generate_instance(self):
        assert self.min_value >= 0 and self.max_value >= self.min_value
        assert self.add_item_prob >= 0 and self.add_item_prob <= 1

        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)
        bids = []

        while len(bids) < self.n_bids:
            bundle_size = np.random.randint(1, self.max_bundle_size + 1)
            bundle = np.random.choice(self.n_items, size=bundle_size, replace=False)
            price = max(values[bundle].sum() + np.random.normal(0, 10), 0)
            complexity = np.random.poisson(lam=5)

            bids.append((bundle.tolist(), price, complexity))

        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price, complexity = bid
            for item in bundle:
                bids_per_item[item].append(i)

        # Facility data generation
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

        # Generate time windows for perishable items
        time_windows = np.random.randint(1, 10, size=self.n_perishable_items).tolist()

        # Generate energy consumption rates for facilities
        energy_consumption = np.random.uniform(0.5, 2.0, size=n_facilities).tolist()

        # Generate raw material availability
        raw_material_availability = np.random.uniform(50, 200, size=self.n_items).tolist()

        # Generate environmental impact data
        environmental_impact = np.random.normal(20, 5, size=n_facilities).tolist()

        # Generate labor costs
        labor_cost = np.random.uniform(10, 50, size=n_facilities).tolist()

        # Generate regional regulations penalty data
        regional_penalties = np.random.uniform(0, 10, size=n_facilities).tolist()

        # Generate vehicle idle time costs
        vehicle_idle_cost = np.random.uniform(1, 10, size=n_facilities).tolist()
        maintenance_budget = np.random.randint(5000, 10000)

        # Generate mutual exclusivity groups for bids
        mutual_exclusivity_groups = []
        for _ in range(self.n_exclusive_groups):
            group_size = random.randint(self.min_group_size, self.max_group_size)
            try:
                exclusive_bids = random.sample(range(len(bids)), group_size)
                mutual_exclusivity_groups.append(exclusive_bids)
            except ValueError:
                pass

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
            "time_windows": time_windows,
            "energy_consumption": energy_consumption,
            "raw_material_availability": raw_material_availability,
            "environmental_impact": environmental_impact,
            "labor_cost": labor_cost,
            "regional_penalties": regional_penalties,
            "vehicle_idle_cost": vehicle_idle_cost,
            "maintenance_budget": maintenance_budget,
            "mutual_exclusivity_groups": mutual_exclusivity_groups # New data
        }

    ################# PySCIPOpt modeling #################
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
        time_windows = instance['time_windows']
        energy_consumption = instance['energy_consumption']
        raw_material_availability = instance['raw_material_availability']
        environmental_impact = instance['environmental_impact']
        labor_cost = instance['labor_cost']
        regional_penalties = instance['regional_penalties']
        vehicle_idle_cost = instance['vehicle_idle_cost']
        maintenance_budget = instance['maintenance_budget']
        mutual_exclusivity_groups = instance['mutual_exclusivity_groups'] # New data

        model = Model("AdvancedResourceAllocationWithComplexities")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(bids)) for j in range(n_facilities)}
        facility_workload = {j: model.addVar(vtype="I", name=f"workload_{j}", lb=0) for j in range(n_facilities)}

        # New perishable good production time variables
        production_time_vars = {i: model.addVar(vtype="C", name=f"prod_time_{i}", lb=0) for i in range(self.n_perishable_items)}

        # New energy consumption variables
        energy_consumption_vars = {j: model.addVar(vtype="C", name=f"energy_{j}", lb=0) for j in range(n_facilities)}

        # New raw material usage variables
        raw_material_usage_vars = {i: model.addVar(vtype="C", name=f"raw_material_{i}", lb=0) for i in range(self.n_items)}

        # New labor cost variables
        labor_cost_vars = {j: model.addVar(vtype="C", name=f"labor_cost_{j}", lb=0) for j in range(n_facilities)}

        # New environmental impact variables
        environmental_impact_vars = {j: model.addVar(vtype="C", name=f"env_impact_{j}", lb=0) for j in range(n_facilities)}

        # New vehicle idle time variables
        vehicle_idle_time_vars = {j: model.addVar(vtype="C", name=f"idle_time_{j}", lb=0) for j in range(n_facilities)}

        # New item won variables from second MILP
        item_vars = {i: model.addVar(vtype="I", name=f"ItemWon_{i}") for i in range(self.n_items)}

        objective_expr = quicksum(price * bid_vars[i] for i, (bundle, price, complexity) in enumerate(bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * x_vars[i, j] for i in range(len(bids)) for j in range(n_facilities)) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(maintenance_cost[j] * facility_workload[j] for j in range(n_facilities)) \
                         - quicksum(complexity * bid_vars[i] for i, (bundle, price, complexity) in enumerate(bids)) \
                         - quicksum(energy_consumption[j] * energy_consumption_vars[j] * self.energy_cost for j in range(n_facilities)) \
                         - quicksum(labor_cost[j] * labor_cost_vars[j] for j in range(n_facilities)) \
                         - quicksum(environmental_impact[j] * environmental_impact_vars[j] for j in range(n_facilities)) \
                         - quicksum(regional_penalties[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(vehicle_idle_cost[j] * vehicle_idle_time_vars[j] for j in range(n_facilities))

        # Constraints: Each item can only be part of one accepted bid
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"Item_{item}")

        # Bid assignment to facility
        for i in range(len(bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i], f"BidFacility_{i}")

        # Facility capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(bids))) <= capacity[j] * y_vars[j], f"FacilityCapacity_{j}")

        # Facility workload constraints
        for j in range(n_facilities):
            model.addCons(facility_workload[j] == quicksum(x_vars[i, j] * bids[i][2] for i in range(len(bids))), f"Workload_{j}")

        # Energy consumption constraints linked to workload
        for j in range(n_facilities):
            model.addCons(energy_consumption_vars[j] == quicksum(x_vars[i, j] * energy_consumption[j] for i in range(len(bids))), f"EnergyConsumption_{j}")

        # Raw material usage constraints
        for i in range(self.n_items):
            model.addCons(raw_material_usage_vars[i] <= raw_material_availability[i], f"RawMaterial_{i}")

        # Labor cost constraints
        for j in range(n_facilities):
            model.addCons(labor_cost_vars[j] <= labor_cost[j], f"LaborCost_{j}")

        # Environmental impact constraints
        for j in range(n_facilities):
            model.addCons(environmental_impact_vars[j] <= environmental_impact[j], f"EnvironmentalImpact_{j}")

        # Vehicle idle time during maintenance constraint
        model.addCons(quicksum(maintenance_cost[j] * vehicle_idle_time_vars[j] for j in range(n_facilities)) <= maintenance_budget, "MaintenanceBudget")

        # Further constraints to introduce complexity
        # Constraints on mutual exclusivity for facilities
        for fac1, fac2 in mutual_exclusivity_pairs:
            model.addCons(y_vars[fac1] + y_vars[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        # Production time constraints for perishable goods
        for i in range(self.n_perishable_items):
            model.addCons(production_time_vars[i] <= time_windows[i], f"PerishableTime_{i}")

        # Mutual exclusivity constraints for groups of bids from second MILP
        for group in mutual_exclusivity_groups:
            model.addCons(quicksum(bid_vars[bid] for bid in group) <= 1, f"ExclusiveGroup_{'_'.join(map(str, group))}")

        # Linking ItemWon variables to bids from second MILP
        for item in range(self.n_items):
            model.addCons(item_vars[item] == quicksum(bid_vars[bid_idx] for bid_idx in bids_per_item[item]), f"LinkItem_{item}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 1874,
        'n_bids': 12,
        'min_value': 1012,
        'max_value': 5000,
        'max_bundle_size': 609,
        'add_item_prob': 0.31,
        'facility_min_count': 2250,
        'facility_max_count': 2810,
        'complexity_mean': 1680,
        'complexity_stddev': 1350,
        'n_exclusive_pairs': 555,
        'n_perishable_items': 750,
        'energy_cost': 0.24,
        ### new parameter code starts here
        'n_exclusive_groups': 10, 
        'min_group_size': 2, 
        'max_group_size': 10
        ### new parameter code ends here
    }

    auction = AdvancedResourceAllocationWithComplexities(parameters, seed=42)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")