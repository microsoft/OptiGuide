import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ManufacturingDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_factories = random.randint(self.min_factories, self.max_factories)
        num_retailers = random.randint(self.min_retailers, self.max_retailers)
        
        # Costs and capacities
        new_product_costs = np.random.randint(50, 200, size=(num_retailers, num_factories))
        manufacturing_costs = np.random.randint(1000, 5000, size=num_factories)
        manufacturing_budget = np.random.randint(100000, 200000)
        
        manufacturing_capacity = np.random.randint(50, 200, size=num_factories)
        demand_penalty = np.random.randint(100, 500)
        
        # Efficiency matrices for handling and transportation lines
        num_lines = 3  # let's consider 3 types of lines
        handle_lines_efficiency = np.random.uniform(0.7, 1.2, size=(num_lines, num_retailers, num_factories))
        handle_usage = np.random.randint(1, 5, size=(num_lines, num_retailers, num_factories))

        res = {
            'num_factories': num_factories,
            'num_retailers': num_retailers,
            'new_product_costs': new_product_costs,
            'manufacturing_costs': manufacturing_costs,
            'manufacturing_budget': manufacturing_budget,
            'manufacturing_capacity': manufacturing_capacity,
            'demand_penalty': demand_penalty,
            'num_lines': num_lines,
            'handle_lines_efficiency': handle_lines_efficiency,
            'handle_usage': handle_usage,
        }
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_factories = instance['num_factories']
        num_retailers = instance['num_retailers']
        new_product_costs = instance['new_product_costs']
        manufacturing_costs = instance['manufacturing_costs']
        manufacturing_budget = instance['manufacturing_budget']
        manufacturing_capacity = instance['manufacturing_capacity']
        demand_penalty = instance['demand_penalty']
        num_lines = instance['num_lines']
        handle_lines_efficiency = instance['handle_lines_efficiency']
        handle_usage = instance['handle_usage']
        
        model = Model("ManufacturingDistributionOptimization")
        
        # Variables
        factory = {j: model.addVar(vtype="B", name=f"factory_{j}") for j in range(num_factories)}
        production = {(i, j): model.addVar(vtype="B", name=f"production_{i}_{j}") for i in range(num_retailers) for j in range(num_factories)}
        demand_miss = {j: model.addVar(vtype="C", name=f"demand_miss_{j}") for j in range(num_factories)}
        
        # Line usage variables
        handle_usable_vars = {(l, i, j): model.addVar(vtype='B', name=f'handle_usable_{l}_{i}_{j}') for l in range(num_lines) for i in range(num_retailers) for j in range(num_factories)}
        
        # Objective function: Minimize total costs including handling efficiency
        total_cost = quicksum(
            production[i, j] * new_product_costs[i, j] * handle_lines_efficiency[l, i, j] * handle_usage[l, i, j] 
            for i in range(num_retailers) for j in range(num_factories) for l in range(num_lines)
        ) + quicksum(
            factory[j] * manufacturing_costs[j] for j in range(num_factories)
        ) + quicksum(
            demand_miss[j] * demand_penalty for j in range(num_factories)
        )
        
        model.setObjective(total_cost, "minimize")
        
        # Constraints
        
        # Each retailer should receive products from at least one factory
        for i in range(num_retailers):
            model.addCons(quicksum(production[i, j] for j in range(num_factories)) >= 1, name=f"retailer_supply_{i}")
        
        # A factory can produce only if it is operational
        for j in range(num_factories):
            for i in range(num_retailers):
                model.addCons(production[i, j] <= factory[j], name=f"factory_production_{i}_{j}")
        
        # Budget constraint on the total manufacturing cost
        model.addCons(quicksum(factory[j] * manufacturing_costs[j] for j in range(num_factories)) <= manufacturing_budget, name="manufacturing_budget_constraint")
        
        # Maximum allowable production volume if operational (Big M Formulation)
        for j in range(num_factories):
            model.addCons(factory[j] <= manufacturing_capacity[j], name=f"max_production_volume_{j}")

        # Line usability constraint based on handling conditions
        for l in range(num_lines):
            for i in range(num_retailers):
                for j in range(num_factories):
                    model.addCons(handle_usable_vars[l, i, j] <= handle_usage[l, i, j], name=f"line_handle_usable_{l}_{i}_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_factories': 37,
        'max_factories': 600,
        'min_retailers': 80,
        'max_retailers': 800,
    }
    
    optimization = ManufacturingDistributionOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")