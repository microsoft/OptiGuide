import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class CapacitatedFacilityLocation:
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
        
        model = Model("CapacitatedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Semi-continuous variables: Operational level for open facilities
        operational_level = {j: model.addVar(vtype="C", lb=0, name=f"OperationalLevel_{j}") for j in range(n_facilities)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        
        # Constraints: demand must be met
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        # Constraints: tightening constraints
        total_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, "TotalDemand")
        
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}")

        # New Constraints: operational level for open facilities
        for j in range(n_facilities):
            model.addCons(operational_level[j] >= self.operational_min_threshold * open_facilities[j], f"OperationalMin_{j}")
            model.addCons(operational_level[j] <= capacities[j] * open_facilities[j], f"OperationalMax_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 300,
        'n_facilities': 50,
        'demand_interval': (40, 288),
        'capacity_interval': (10, 161),
        'fixed_cost_scale_interval': (200, 222),
        'fixed_cost_cste_interval': (0, 91),
        'ratio': 5.0,
        'continuous_assignment': 2,
        'operational_min_threshold': 80,
    }

    facility_location = CapacitatedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")