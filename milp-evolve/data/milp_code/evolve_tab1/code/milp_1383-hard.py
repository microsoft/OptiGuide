import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class DeterministicCapacitatedFacilityLocation:
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
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.fixed_cost * np.ones(self.n_facilities)
        transportation_costs = self.generate_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        # New instance data for multi-type facilities
        facility_types = self.facility_types
        for facility_type in facility_types:
            res[f'capacities_{facility_type}'] = self.randint(self.n_facilities, self.capacity_interval)
            res[f'fixed_costs_{facility_type}'] = self.fixed_cost * np.ones(self.n_facilities)
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("DeterministicCapacitatedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        unmet_demand = {i: model.addVar(vtype="C", name=f"Unmet_{i}", ub=demands[i]) for i in range(n_customers)}

        # Objective: minimize the total cost
        penalty_cost = 1000  # high penalty cost for unmet demand
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + quicksum(penalty_cost * unmet_demand[i] for i in range(n_customers))
        
        # Constraints: demand must be met or counted as unmet demand
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) + unmet_demand[i] >= 1, f"Demand_{i}")
        
        # Constraints: simple capacity limits
        for j in range(n_facilities):
            capacity_expr = quicksum(serve[i, j] * demands[i] for i in range(n_customers))
            model.addCons(capacity_expr <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        # Example addition: ensuring 80% of demand is met within a specific cost
        service_level = 0.8
        service_cost_limit = 3000
        for i in range(n_customers):
            model.addCons(quicksum(transportation_costs[i, j] * serve[i, j] for j in range(n_facilities)) <= service_cost_limit + (1 - service_level), f"ServiceLevel_{i}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 3000,
        'n_facilities': 5,
        'demand_interval': (0, 4),
        'capacity_interval': (27, 432),
        'fixed_cost': 1500,
        'ratio': 393.75,
        'facility_types': ['type1', 'type2']
    }

    facility_location = DeterministicCapacitatedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")