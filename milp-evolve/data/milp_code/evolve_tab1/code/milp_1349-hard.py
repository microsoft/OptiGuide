import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class StochasticCapacitatedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2
        )
        return costs

    def generate_instance(self):
        demands_mu = self.randint(self.n_customers, self.demand_interval)
        demands_sigma = self.demand_uncertainty_scale * np.ones(self.n_customers)  # Simplified demand uncertainty
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.fixed_cost * np.ones(self.n_facilities)  # Simplified fixed costs
        transportation_costs = self.generate_transportation_costs() * demands_mu[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands_mu) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands_mu': demands_mu,
            'demands_sigma': demands_sigma,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands_mu = instance['demands_mu']
        demands_sigma = instance['demands_sigma']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        
        n_customers = len(demands_mu)
        n_facilities = len(capacities)
        
        model = Model("StochasticCapacitatedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        
        # Constraints: demand must be met
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")
        
        # Constraints: robust capacity limits
        for j in range(n_facilities):
            robust_capacity_expr = quicksum((serve[i, j] * (demands_mu[i] + self.beta * demands_sigma[i])) for i in range(n_customers))
            model.addCons(robust_capacity_expr <= capacities[j] * open_facilities[j], f"RobustCapacity_{j}")
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 500,
        'n_facilities': 5,
        'demand_interval': (0, 4),
        'capacity_interval': (3, 48),
        'fixed_cost': 750,
        'ratio': 11.25,
        'demand_uncertainty_scale': 0.59,
        'beta': 1.47,
    }

    facility_location = StochasticCapacitatedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")