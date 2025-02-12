import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods >= self.n_facilities
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        assert self.n_tiers > 1

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_neighborhoods))
        base_capacity = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)
        capacity_costs = {f: np.random.uniform(0.5, 1.5, self.n_tiers) for f in range(self.n_facilities)}

        financial_rewards = np.random.uniform(10, 100, self.n_neighborhoods)

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "base_capacity": base_capacity,
            "capacity_costs": capacity_costs,
            "financial_rewards": financial_rewards,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        base_capacity = instance['base_capacity']
        capacity_costs = instance['capacity_costs']
        financial_rewards = instance['financial_rewards']
        
        model = Model("AdvancedFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        tier_vars = {(f, t): model.addVar(vtype="C", name=f"Facility_{f}_Tier_{t}") for f in range(n_facilities) for t in range(self.n_tiers)}
        
        # Objective: maximize financial rewards from treated neighborhoods minus costs (fixed and transport)
        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(capacity_costs[f][t] * tier_vars[f, t] for f in range(n_facilities) for t in range(self.n_tiers)),
            "maximize"
        )

        # Constraints: Each neighborhood is served by exactly one facility
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        # Constraints: Only open facilities can serve neighborhoods
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")
        
        # Constraints: Facilities cannot exceed their tiered capacities
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= 
                          quicksum(tier_vars[f, t] for t in range(self.n_tiers)) * base_capacity[f], f"Facility_{f}_Capacity")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 25,
        'n_neighborhoods': 200,
        'min_fixed_cost': 1125,
        'max_fixed_cost': 1875,
        'min_transport_cost': 675,
        'max_transport_cost': 1500,
        'min_capacity': 330,
        'max_capacity': 3000,
        'n_tiers': 2,
    }
    ### updated parameter code ends here

    location_optimizer = FacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")