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
        assert self.n_facilities > 0 and self.n_neighborhoods > 0
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.normal((self.min_transport_cost + self.max_transport_cost) / 2, 
                                           (self.max_transport_cost - self.min_transport_cost) / 6, 
                                           (self.n_facilities, self.n_neighborhoods))
        transport_costs = np.clip(transport_costs, self.min_transport_cost, self.max_transport_cost).astype(int)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)
        
        # Additional data for semi-continuous transport costs
        semi_continuous_min = np.random.randint(self.min_semi_continuous, self.max_semi_continuous + 1, (self.n_facilities, self.n_neighborhoods))

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "semi_continuous_min": semi_continuous_min,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        semi_continuous_min = instance['semi_continuous_min']
        
        model = Model("FacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        
        # Semi-continuous transport cost variables
        transport_cost_vars = {(f, n): model.addVar(vtype="C", lb=0, ub=self.max_transport_cost, name=f"TransCost_{f}_Neighborhood_{n}") 
                               for f in range(n_facilities) for n in range(n_neighborhoods)}

        # Objective: minimize the total cost (fixed + transport)
        model.setObjective(quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) +
                           quicksum(transport_cost_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)), "minimize")
        
        # Constraints: Each neighborhood is served by exactly one facility (Set Packing)
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        # Constraints: Only open facilities can serve neighborhoods
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")
        
        # Constraints: Facilities cannot serve more neighborhoods than their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")
        
        # Constraints: Semi-continuous transport costs
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(transport_cost_vars[f, n] >= semi_continuous_min[f][n] * allocation_vars[f, n], f"SemiCont_{f}_Neighborhood_{n}")
                model.addCons(transport_cost_vars[f, n] <= self.max_transport_cost * allocation_vars[f, n], f"SemiContUpper_{f}_Neighborhood_{n}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 75,
        'n_neighborhoods': 75,
        'min_fixed_cost': 2250,
        'max_fixed_cost': 10000,
        'min_transport_cost': 101,
        'max_transport_cost': 2000,
        'min_capacity': 126,
        'max_capacity': 3000,
        'min_semi_continuous': 375,
        'max_semi_continuous': 750,
    }
    
    location_optimizer = FacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")