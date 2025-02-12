import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def unit_transport_costs(self):
        return np.random.rand(self.n_customers, self.n_facilities) * 10

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.randint(self.n_facilities, self.fixed_cost_interval)
        travel_times = self.unit_transport_costs()

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'travel_times': travel_times
        }
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        travel_times = instance['travel_times']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("SimplifiedFacilityLocation")
        
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(travel_times[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) == 1, f"Demand_{i}")
        
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 50,
        'n_facilities': 150,
        'demand_interval': (7, 37),
        'capacity_interval': (300, 600),
        'fixed_cost_interval': (600, 1200),
        'ratio': 20.0,
    }

    new_facility_location = SimplifiedFacilityLocation(parameters, seed=seed)
    instance = new_facility_location.generate_instance()
    solve_status, solve_time = new_facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")