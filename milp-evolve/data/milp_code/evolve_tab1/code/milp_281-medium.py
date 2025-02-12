import random
import time
import numpy as np
import networkx as nx  # We will use networkx to generate more complex data
from pyscipopt import Model, quicksum

class SupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    # Data Generation
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_transportation_costs(self):
        return np.random.rand(self.n_customers, self.n_suppliers) * self.cost_scale

    def generate_transportation_emissions(self):
        return np.random.rand(self.n_customers, self.n_suppliers) * self.emission_scale

    def generate_suppliers_fair_labor_practices(self):
        return np.random.randint(0, 2, self.n_suppliers)

    def generate_suppliers_geographical_zones(self):
        return np.random.randint(0, len(self.geographical_zones), self.n_suppliers)

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        supplier_capacities = self.randint(self.n_suppliers, self.capacity_interval)
        fixed_costs = self.randint(self.n_suppliers, self.fixed_cost_interval)
        transport_costs = self.generate_transportation_costs()
        transport_emissions = self.generate_transportation_emissions()
        fair_labor_flags = self.generate_suppliers_fair_labor_practices()
        geographical_zones = self.generate_suppliers_geographical_zones()

        # Realistic dataset for diversified suppliers
        res = {
            'demands': demands,
            'supplier_capacities': supplier_capacities,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs,
            'transport_emissions': transport_emissions,
            'fair_labor_flags': fair_labor_flags,
            'geographical_zones': geographical_zones,
        }

        return res

    # MILP Solver
    def solve(self, instance):
        demands = instance['demands']
        supplier_capacities = instance['supplier_capacities']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        transport_emissions = instance['transport_emissions']
        fair_labor_flags = instance['fair_labor_flags']
        geographical_zones = instance['geographical_zones']
        
        n_customers = len(demands)
        n_suppliers = len(supplier_capacities)
        M = 1e6  # Big M constant
        
        model = Model("SupplyChainOptimization")
        
        # Decision variables
        open_suppliers = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_suppliers)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(n_customers) for j in range(n_suppliers)}

        # Objective: minimize the total cost including emissions and penalties for not adhering to fair practices
        objective_expr = quicksum(fixed_costs[j] * open_suppliers[j] for j in range(n_suppliers)) + \
                         quicksum(transport_costs[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_suppliers)) + \
                         quicksum(transport_emissions[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_suppliers)) + \
                         quicksum((1 - fair_labor_flags[j]) * self.fair_labor_penalty * open_suppliers[j] for j in range(n_suppliers))

        # Constraints
        # Demand satisfaction constraints
        for i in range(n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(n_suppliers)) == demands[i], f"Demand_{i}")

        # Supplier capacity constraints
        for j in range(n_suppliers):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) <= supplier_capacities[j], f"SupplierCapacity_{j}")
            for i in range(n_customers):
                model.addCons(flow[i, j] <= M * open_suppliers[j], f"BigM_{i}_{j}")

        # Supplier Fair Labor Adherence
        total_open_suppliers = quicksum(open_suppliers[j] for j in range(n_suppliers))
        fair_labor_suppliers = quicksum(fair_labor_flags[j] * open_suppliers[j] for j in range(n_suppliers))
        model.addCons(fair_labor_suppliers >= self.min_fair_labor_percentage * total_open_suppliers, "FairLaborConstraint")

        # Geographical Diversity Constraint: Ensure at least n geographical zones are represented
        for zone in range(len(self.geographical_zones)):
            count_per_zone = quicksum(open_suppliers[j] for j in range(n_suppliers) if geographical_zones[j] == zone)
            model.addCons(count_per_zone >= self.min_suppliers_per_zone, f"GeographicalDiversity_{zone}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 123
    parameters = {
        'n_customers': 20,
        'n_suppliers': 1000,
        'demand_interval': (500, 2000),
        'capacity_interval': (80, 400),
        'fixed_cost_interval': (5000, 10000),
        'cost_scale': 40,
        'emission_scale': 0.62,
        'fair_labor_penalty': 3000,
        'min_fair_labor_percentage': 0.79,
        'geographical_zones': ('Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5'),
        'min_suppliers_per_zone': 20,
    }

    supply_chain_optimization = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_chain_optimization.generate_instance()
    solve_status, solve_time = supply_chain_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")