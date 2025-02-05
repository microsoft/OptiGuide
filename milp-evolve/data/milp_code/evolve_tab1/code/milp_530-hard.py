import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HealthcareFacilityDeploymentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_healthcare_facilities(self):
        n_locations = np.random.randint(self.min_facilities, self.max_facilities)
        facilities = np.random.choice(range(2000), size=n_locations, replace=False)  # Random unique healthcare facility IDs
        return facilities

    def generate_regions(self, n_facilities):
        n_regions = np.random.randint(self.min_regions, self.max_regions)
        regions = np.random.choice(range(3000, 4000), size=n_regions, replace=False)
        demands = {region: np.random.randint(self.min_demand, self.max_demand) for region in regions}
        return regions, demands

    def generate_assignment_costs(self, facilities, regions):
        costs = {(f, r): np.random.uniform(self.min_assign_cost, self.max_assign_cost) for f in facilities for r in regions}
        return costs

    def generate_operational_capacities(self, facilities):
        capacities = {f: np.random.randint(self.min_capacity, self.max_capacity) for f in facilities}
        return capacities

    def get_instance(self):
        facilities = self.generate_healthcare_facilities()
        regions, demands = self.generate_regions(len(facilities))
        assign_costs = self.generate_assignment_costs(facilities, regions)
        capacities = self.generate_operational_capacities(facilities)
        
        install_costs = {f: np.random.uniform(self.min_install_cost, self.max_install_cost) for f in facilities}
        operational_costs = {f: np.random.uniform(self.min_operational_cost, self.max_operational_cost) for f in facilities}
        extra_costs = {f: np.random.uniform(self.min_extra_cost, self.max_extra_cost) for f in facilities}
        
        return {
            'facilities': facilities,
            'regions': regions,
            'assign_costs': assign_costs,
            'capacities': capacities,
            'demands': demands,
            'install_costs': install_costs,
            'operational_costs': operational_costs,
            'extra_costs': extra_costs,
        }

    def solve(self, instance):
        facilities = instance['facilities']
        regions = instance['regions']
        assign_costs = instance['assign_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        install_costs = instance['install_costs']
        operational_costs = instance['operational_costs']
        extra_costs = instance['extra_costs']

        model = Model("HealthcareFacilityDeploymentOptimization")

        NewHealthcareFacility_vars = {f: model.addVar(vtype="B", name=f"Facility_Loc_{f}") for f in facilities}
        ZonePatientAssignment_vars = {(f, r): model.addVar(vtype="B", name=f"PatientAssign_{f}_{r}") for f in facilities for r in regions}
        CapacityUsage_vars = {(f, r): model.addVar(vtype="C", name=f"CapacityUsage_{f}_{r}") for f in facilities for r in regions}
        MaximumFacilityCapacity_vars = {f: model.addVar(vtype="C", name=f"MaxFacilityCapacity_{f}") for f in facilities}
        HealthcareArrangement_vars = {f: model.addVar(vtype="C", name=f"TempArrangement_{f}") for f in facilities}

        # Objective function
        total_cost = quicksum(NewHealthcareFacility_vars[f] * install_costs[f] for f in facilities)
        total_cost += quicksum(ZonePatientAssignment_vars[f, r] * assign_costs[f, r] for f in facilities for r in regions)
        total_cost += quicksum(MaximumFacilityCapacity_vars[f] * operational_costs[f] for f in facilities)
        total_cost += quicksum(HealthcareArrangement_vars[f] * extra_costs[f] for f in facilities)
        
        model.setObjective(total_cost, "minimize")

        # Constraints
        for f in facilities:
            model.addCons(
                quicksum(CapacityUsage_vars[f, r] for r in regions) <= capacities[f] * NewHealthcareFacility_vars[f],
                name=f"FacilityCapacity_{f}"
            )

        for r in regions:
            model.addCons(
                quicksum(CapacityUsage_vars[f, r] for f in facilities) >= demands[r],
                name=f"RegionDemand_{r}"
            )

        for f in facilities:
            model.addCons(
                MaximumFacilityCapacity_vars[f] <= capacities[f],
                name=f"MaxCapacityLimit_{f}"
            )

        for f in facilities:
            model.addCons(
                HealthcareArrangement_vars[f] <= capacities[f] * 0.2,  # Assume temporary arrangement is limited to 20% of the capacity
                name=f"TempArrangementLimit_{f}"
            )

        model.setParam('limits/time', 10 * 60)  # Set a time limit of 10 minutes for solving
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facilities': 18,
        'max_facilities': 300,
        'min_regions': 105,
        'max_regions': 150,
        'min_demand': 350,
        'max_demand': 600,
        'min_assign_cost': 21.0,
        'max_assign_cost': 375.0,
        'min_capacity': 750,
        'max_capacity': 1000,
        'min_install_cost': 25000,
        'max_install_cost': 200000,
        'min_operational_cost': 3000,
        'max_operational_cost': 5000,
        'min_extra_cost': 400,
        'max_extra_cost': 3000,
    }

    optimizer = HealthcareFacilityDeploymentOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")