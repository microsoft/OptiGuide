import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_regions > 0
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_region_benefit >= 0 and self.max_region_benefit >= self.min_region_benefit
        assert self.min_facility_cap > 0 and self.max_facility_cap >= self.min_facility_cap

        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        region_benefits = np.random.randint(self.min_region_benefit, self.max_region_benefit + 1, (self.n_facilities, self.n_regions))
        capacities = np.random.randint(self.min_facility_cap, self.max_facility_cap + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_regions)
        
        return {
            "facility_costs": facility_costs,
            "region_benefits": region_benefits,
            "capacities": capacities,
            "demands": demands,
        }

    def solve(self, instance):
        facility_costs = instance['facility_costs']
        region_benefits = instance['region_benefits']
        capacities = instance['capacities']
        demands = instance['demands']

        model = Model("FacilityLocationOptimization")
        n_facilities = len(facility_costs)
        n_regions = len(region_benefits[0])

        # Decision variables
        open_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        assign_vars = {(f, r): model.addVar(vtype="C", name=f"Assign_{f}_{r}") for f in range(n_facilities) for r in range(n_regions)}

        # Objective: maximize the total benefit minus the facility costs.
        model.setObjective(
            quicksum(region_benefits[f, r] * assign_vars[f, r] for f in range(n_facilities) for r in range(n_regions)) -
            quicksum(facility_costs[f] * open_vars[f] for f in range(n_facilities)),
            "maximize"
        )

        # Constraints: Each region's demand must be met by the facilities.
        for r in range(n_regions):
            model.addCons(quicksum(assign_vars[f, r] for f in range(n_facilities)) == demands[r], f"Region_{r}_Demand")
        
        # Constraints: Only open facilities can serve regions.
        for f in range(n_facilities):
            for r in range(n_regions):
                model.addCons(assign_vars[f, r] <= demands[r] * open_vars[f], f"Facility_{f}_Serve_{r}")
        
        # Constraints: Facilities cannot exceed their capacities.
        for f in range(n_facilities):
            model.addCons(quicksum(assign_vars[f, r] for r in range(n_regions)) <= capacities[f], f"Facility_{f}_Capacity")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 175,
        'n_regions': 75,
        'min_region_benefit': 300,
        'max_region_benefit': 1000,
        'min_facility_cost': 750,
        'max_facility_cost': 3000,
        'min_facility_cap': 1500,
        'max_facility_cap': 3000,
    }

    facility_optimizer = FacilityLocationOptimization(parameters, seed=seed)
    instance = facility_optimizer.generate_instance()
    solve_status, solve_time, objective_value = facility_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")