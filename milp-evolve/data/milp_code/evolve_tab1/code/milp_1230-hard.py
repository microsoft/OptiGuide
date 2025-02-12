import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

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
    
    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_facilities, self.fixed_cost_cste_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("SimplifiedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="C" if self.continuous_assignment else "B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        model.setObjective(objective_expr, "minimize")
        
        # Constraints: demand must be met (simplified)
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 0.5, f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        # Constraints: total capacity meets total demand
        total_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, "TotalDemand")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 100,
        'n_facilities': 300,
        'demand_interval': (30, 216),
        'capacity_interval': (50, 805),
        'fixed_cost_scale_interval': (500, 555),
        'fixed_cost_cste_interval': (0, 91),
        'ratio': 25.0,
        'continuous_assignment': 5,
    }

    facility_location = SimplifiedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")