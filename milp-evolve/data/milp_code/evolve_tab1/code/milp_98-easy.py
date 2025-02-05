import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    # Data Generation
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        return np.random.rand(self.n_customers, self.n_facilities) * self.transport_cost_scale

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.randint(self.n_facilities, self.fixed_cost_interval)
        transport_costs = self.unit_transportation_costs()

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs
        }
        return res

    # MILP Solver
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("SimplifiedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transport_costs[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_facilities))
        
        # Flow conservation constraints
        for i in range(n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(n_facilities)) == demands[i], f"Demand_{i}")
        
        # Flow capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 50,
        'n_facilities': 450,
        'demand_interval': (140, 700),
        'capacity_interval': (350, 1400),
        'fixed_cost_interval': (350, 1400),
        'transport_cost_scale': 90.0,
    }
    
    simplified_facility_location = SimplifiedFacilityLocation(parameters, seed=seed)
    instance = simplified_facility_location.generate_instance()
    solve_status, solve_time = simplified_facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")