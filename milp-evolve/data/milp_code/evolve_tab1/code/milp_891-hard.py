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

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.normal(self.mean_transport_cost, self.stddev_transport_cost, (self.n_facilities, self.n_neighborhoods)).astype(int)
        transport_costs = np.clip(transport_costs, self.min_transport_cost, self.max_transport_cost)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)
        social_benefits = np.random.uniform(10, 100, self.n_neighborhoods)
        upgrade_costs = np.random.uniform(self.min_upgrade_cost, self.max_upgrade_cost, self.n_facilities)

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "social_benefits": social_benefits,
            "upgrade_costs": upgrade_costs,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        social_benefits = instance['social_benefits']
        upgrade_costs = instance['upgrade_costs']
        
        model = Model("EnhancedFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        upgrade_vars = {f: model.addVar(vtype="B", name=f"Upgrade_{f}") for f in range(n_facilities)}
        
        # Objective: maximize social benefits from treated neighborhoods minus costs (fixed, transport, and upgrade)
        model.setObjective(
            quicksum(social_benefits[n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(upgrade_costs[f] * upgrade_vars[f] for f in range(n_facilities)),
            "maximize"
        )
        
        # Constraints: Each neighborhood is served by exactly one facility
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        # Constraints: Only open facilities can serve neighborhoods
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")
                
        # Constraints: Facilities cannot exceed their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")
        
        # Big M constraints: Upgrade variable constraints
        M = max(capacities)  # Big M value, can be any sufficiently large number
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= M * upgrade_vars[f], f"Facility_{f}_Upgrade")
            model.addCons(upgrade_vars[f] <= facility_vars[f], f"Facility_{f}_Upgrade_activation")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 9,
        'n_neighborhoods': 1125,
        'min_fixed_cost': 75,
        'max_fixed_cost': 468,
        'mean_transport_cost': 393,
        'stddev_transport_cost': 11,
        'min_transport_cost': 253,
        'max_transport_cost': 1687,
        'min_capacity': 110,
        'max_capacity': 3000,
        'min_upgrade_cost': 75,
        'max_upgrade_cost': 1000,
    }

    location_optimizer = FacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")