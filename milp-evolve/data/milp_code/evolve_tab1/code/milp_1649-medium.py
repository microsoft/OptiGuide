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

        # Generating vehicle types and fuel efficiency data
        vehicle_types = self.n_vehicle_types
        fuel_efficiency = np.random.uniform(self.efficiency_interval[0], self.efficiency_interval[1], vehicle_types)

        # Generate random cliques
        cliques = [sorted(random.sample(range(self.n_facilities), self.clique_size)) for _ in range(self.n_cliques)]
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'fuel_efficiency': fuel_efficiency,
            'cliques': cliques
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        fuel_efficiency = instance['fuel_efficiency']
        cliques = instance['cliques']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        n_vehicle_types = self.n_vehicle_types
        
        model = Model("CapacitatedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        distance_traveled = {j: model.addVar(vtype="C", name=f"Distance_{j}") for j in range(n_facilities)}

        # Objective: minimize the total cost
        fuel_costs = quicksum(distance_traveled[j] / fuel_efficiency[j % n_vehicle_types] for j in range(n_facilities))
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + fuel_costs

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

        # New Constraints: compute distance traveled for fuel cost calculation
        for j in range(n_facilities):
            model.addCons(distance_traveled[j] == quicksum(transportation_costs[i, j] for i in range(n_customers)), f"DistanceCalculation_{j}")
        
        # New Constraints: Clique constraints
        for k, clique in enumerate(cliques):
            model.addCons(quicksum(open_facilities[j] for j in clique) <= self.clique_size - 1, f"Clique_{k}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 250,
        'n_facilities': 75,
        'demand_interval': (21, 189),
        'capacity_interval': (30, 483),
        'fixed_cost_scale_interval': (2500, 2775),
        'fixed_cost_cste_interval': (0, 2),
        'ratio': 75.0,
        'continuous_assignment': 0,
        'n_vehicle_types': 20,
        'efficiency_interval': (225, 675),
        'n_cliques': 5,
        'clique_size': 5,
    }

    facility_location = CapacitatedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")