import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum


class FactoryProduction:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # Randomly generate production costs, selling prices, demand, and resource availability
        production_costs = np.random.randint(1, self.max_production_cost, size=self.n_products)
        selling_prices = production_costs + np.random.randint(1, self.max_price_margin, size=self.n_products)
        demands = np.random.randint(1, self.max_demand, size=self.n_products)

        # Randomly generate resource usage matrix (resources x products)
        resource_usage = np.random.randint(1, self.max_resource_usage, size=(self.n_resources, self.n_products))
        resources_available = np.random.randint(self.min_resources, self.max_resources, size=self.n_resources)

        # Generate assembly links: for product a, b such that b requires output of a to produce
        assembly_links = []
        for _ in range(self.n_assembly_links):
            a, b = np.random.choice(self.n_products, size=2, replace=False)
            assembly_links.append((a, b))

        return {
            'production_costs': production_costs,
            'selling_prices': selling_prices,
            'demands': demands,
            'resource_usage': resource_usage,
            'resources_available': resources_available,
            'assembly_links': assembly_links
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        production_costs = instance['production_costs']
        selling_prices = instance['selling_prices']
        demands = instance['demands']
        resource_usage = instance['resource_usage']
        resources_available = instance['resources_available']
        assembly_links = instance['assembly_links']

        model = Model("FactoryProduction")
        var_names = {}

        # Create variables for production levels of each product
        for j in range(self.n_products):
            var_names[j] = model.addVar(vtype="I", name=f"prod_level_{j}")

        # Add constraints for resource usage
        for r in range(self.n_resources):
            model.addCons(quicksum(var_names[j] * resource_usage[r, j] for j in range(self.n_products)) <= resources_available[r], f"resource_{r}_usage")

        # Add constraints for demand satisfaction
        for j in range(self.n_products):
            model.addCons(var_names[j] <= demands[j], f"demand_{j}")

        # Add constraints for assembly links
        for (a, b) in assembly_links:
            model.addCons(var_names[b] <= var_names[a], f"assembly_{a}_to_{b}")

        # Set objective: Maximize total profit
        objective_expr = quicksum((selling_prices[j] - production_costs[j]) * var_names[j] for j in range(self.n_products))
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_products': 1250,
        'n_resources': 50,
        'n_assembly_links': 10,
        'max_production_cost': 1000,
        'max_price_margin': 450,
        'max_demand': 100,
        'max_resource_usage': 350,
        'min_resources': 500,
        'max_resources': 1000,
    }

    factory_production_problem = FactoryProduction(parameters, seed=seed)
    instance = factory_production_problem.generate_instance()
    solve_status, solve_time = factory_production_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")