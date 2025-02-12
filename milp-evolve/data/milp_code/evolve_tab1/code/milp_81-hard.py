import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NetworkFlowFacilityLocation:
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
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2
        )
        return costs

    def generate_network_flow_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_facilities, self.fixed_cost_cste_interval)
        )
        travel_times = self.unit_transportation_costs() * demands[:, np.newaxis]

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'travel_times': travel_times
        }

        return res

    # MILP Solver
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        travel_times = instance['travel_times']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("NetworkFlowFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(travel_times[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_facilities))
        
        # Flow conservation constraints
        for i in range(n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(n_facilities)) == demands[i], f"Demand_{i}")
        
        # Flow capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        # Constraints: tightening constraints
        total_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, "TotalDemand")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 100,
        'n_facilities': 300,
        'demand_interval': (20, 144),
        'capacity_interval': (30, 483),
        'fixed_cost_scale_interval': (500, 555),
        'fixed_cost_cste_interval': (0, 91),
    }
    
    network_flow_facility_location = NetworkFlowFacilityLocation(parameters, seed=seed)
    instance = network_flow_facility_location.generate_network_flow_instance()
    solve_status, solve_time = network_flow_facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")