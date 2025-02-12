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

        # Stochastic demand variation to increase problem difficulty
        stochastic_demand_factor = np.random.normal(1, self.stochastic_variation, self.n_customers)
        stochastic_demands = np.round(demands * stochastic_demand_factor)

        res = {
            'demands': stochastic_demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'fuel_efficiency': fuel_efficiency,
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        fuel_efficiency = instance['fuel_efficiency']

        n_customers = len(demands)
        n_facilities = len(capacities)
        n_vehicle_types = self.n_vehicle_types

        model = Model("CapacitatedFacilityLocation")

        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Auxiliary Variables for dynamic cost calculation
        dynamic_fuel_costs = {j: model.addVar(vtype="C", name=f"DynamicFuel_{j}") for j in range(n_facilities)}

        # Objective: Minimize the total cost with dynamic fuel cost calculation
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + quicksum(dynamic_fuel_costs[j] for j in range(n_facilities))

        # Constraints: Demand must be met
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")

        # Constraints: Capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        # Constraint: Total facility capacity must cover total stochastic demand
        total_stochastic_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_stochastic_demand, "TotalStochasticDemand")

        # Constraint: Forcing assignments to open facilities only
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}")

        # New Constraints: Dynamic fuel cost calculation based on distance variability
        for j in range(n_facilities):
            model.addCons(dynamic_fuel_costs[j] == quicksum(transportation_costs[i, j] * (1 + 0.1 * np.sqrt(demands[i])) for i in range(n_customers)) / fuel_efficiency[j % n_vehicle_types], f"DynamicFuelCalculation_{j}")

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
        'stochastic_variation': 0.15,
    }

    facility_location = CapacitatedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")