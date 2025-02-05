import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HealthcareFacilityOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods > 0
        assert self.min_travel_cost >= 0 and self.max_travel_cost >= self.min_travel_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_hospital_capacity > 0 and self.max_hospital_capacity >= self.min_hospital_capacity
        
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        travel_costs = np.random.randint(self.min_travel_cost, self.max_travel_cost + 1, (self.n_facilities, self.n_neighborhoods))
        capacities = np.random.randint(self.min_hospital_capacity, self.max_hospital_capacity + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_neighborhoods)
        
        return {
            "facility_costs": facility_costs,
            "travel_costs": travel_costs,
            "capacities": capacities,
            "demands": demands
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        travel_costs = instance['travel_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        
        model = Model("HealthcareFacilityOptimization")
        n_facilities = len(facility_costs)
        n_neighborhoods = len(travel_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        assign_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighbor_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        
        # Objective: minimize the total cost
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(travel_costs[f, n] * assign_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)), "minimize"
        )
        
        # Constraints: Each neighborhood demand is met by exactly one facility
        for n in range(n_neighborhoods):
            model.addCons(quicksum(assign_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Demand")
        
        # Constraints: Only open facilities can serve neighborhoods
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(assign_vars[f, n] <= facility_vars[f], f"Facility_{f}_Serve_{n}")
        
        # Constraints: Facilities cannot exceed their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(demands[n] * assign_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 40,
        'n_neighborhoods': 160,
        'min_travel_cost': 25,
        'max_travel_cost': 2800,
        'min_facility_cost': 3000,
        'max_facility_cost': 4000,
        'min_hospital_capacity': 300,
        'max_hospital_capacity': 2700,
    }

    healthcare_optimizer = HealthcareFacilityOptimization(parameters, seed=42)
    instance = healthcare_optimizer.generate_instance()
    solve_status, solve_time, objective_value = healthcare_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")