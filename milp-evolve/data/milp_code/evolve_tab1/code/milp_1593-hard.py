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
        
        emissions = self.randint(self.n_customers, self.emission_interval)
        emission_limits = self.randint(self.n_facilities, self.emission_limit_interval)
        
        # New data for integrating maintenance states and renewable energy
        maintenance_periods = self.randint(self.n_facilities, self.maintenance_interval)
        renewable_energy_costs = self.randint(self.n_facilities, self.renewable_energy_interval)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'emissions': emissions,
            'emission_limits': emission_limits,
            'maintenance_periods': maintenance_periods,
            'renewable_energy_costs': renewable_energy_costs
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        emissions = instance['emissions']
        emission_limits = instance['emission_limits']
        maintenance_periods = instance['maintenance_periods']
        renewable_energy_costs = instance['renewable_energy_costs']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("CapacitatedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        
        # New decision variables
        maintenance_status = {(j, p): model.addVar(vtype="B", name=f"Maintenance_{j}_{p}") for j in range(n_facilities) for p in range(maintenance_periods[j])}
        renewable_energy_usage = {j: model.addVar(vtype="B", name=f"RenewableEnergy_{j}") for j in range(n_facilities)}
        
        # Objective: minimize the total cost including emission penalties and renewable energy costs
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + \
                         quicksum(emissions[i] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + \
                         quicksum(renewable_energy_costs[j] * renewable_energy_usage[j] for j in range(n_facilities))

        model.setObjective(objective_expr, "minimize")

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

        # New Constraint: Emission limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * emissions[i] for i in range(n_customers)) <= emission_limits[j] * open_facilities[j], f"Emission_{j}")

        # New Constraints: Maintenance periods
        for j in range(n_facilities):
            for p in range(maintenance_periods[j]):
                model.addCons(quicksum(serve[i, j] for i in range(n_customers)) <= (1 - maintenance_status[j, p]) * self.M, f"Maintenance_{j}_{p}")

        # New Constraints: Renewable energy usage
        for j in range(n_facilities):
            model.addCons(renewable_energy_usage[j] >= open_facilities[j] * self.renewable_energy_threshold, f"RenewableEnergyUsage_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 500,
        'n_facilities': 50,
        'demand_interval': (35, 252),
        'capacity_interval': (90, 1449),
        'fixed_cost_scale_interval': (50, 55),
        'fixed_cost_cste_interval': (0, 45),
        'ratio': 3.75,
        'continuous_assignment': 10,
        'emission_interval': (3, 30),
        'emission_limit_interval': (250, 500),
        'maintenance_interval': (0, 3),
        'renewable_energy_interval': (500, 3000),
        'M': 1000000.0,
        'renewable_energy_threshold': 0.24,
    }

    facility_location = CapacitatedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")