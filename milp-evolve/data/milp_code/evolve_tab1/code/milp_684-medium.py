import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_manufacturing_hubs = random.randint(self.min_manufacturing_hubs, self.max_manufacturing_hubs)
        num_distribution_centers = random.randint(self.min_distribution_centers, self.max_distribution_centers)

        # Cost matrices
        transportation_costs = np.random.randint(50, 300, size=(num_distribution_centers, num_manufacturing_hubs))
        fixed_costs = np.random.randint(1000, 5000, size=num_manufacturing_hubs)

        # Distribution center demands
        distribution_demand = np.random.randint(100, 500, size=num_distribution_centers)

        # Manufacturing hub capacities
        manufacturing_capacity = np.random.randint(1000, 5000, size=num_manufacturing_hubs)

        res = {
            'num_manufacturing_hubs': num_manufacturing_hubs,
            'num_distribution_centers': num_distribution_centers,
            'transportation_costs': transportation_costs,
            'fixed_costs': fixed_costs,
            'distribution_demand': distribution_demand,
            'manufacturing_capacity': manufacturing_capacity,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_manufacturing_hubs = instance['num_manufacturing_hubs']
        num_distribution_centers = instance['num_distribution_centers']
        transportation_costs = instance['transportation_costs']
        fixed_costs = instance['fixed_costs']
        distribution_demand = instance['distribution_demand']
        manufacturing_capacity = instance['manufacturing_capacity']

        model = Model("SupplyChainNetworkOptimization")

        # Variables
        manufacturing_hub = {j: model.addVar(vtype="B", name=f"manufacturing_hub_{j}") for j in range(num_manufacturing_hubs)}
        transportation = {(i, j): model.addVar(vtype="B", name=f"transportation_{i}_{j}") for i in range(num_distribution_centers) for j in range(num_manufacturing_hubs)}

        # Objective function: Minimize total costs
        total_cost = quicksum(transportation[i, j] * transportation_costs[i, j] for i in range(num_distribution_centers) for j in range(num_manufacturing_hubs)) + \
                     quicksum(manufacturing_hub[j] * fixed_costs[j] for j in range(num_manufacturing_hubs))
        model.setObjective(total_cost, "minimize")

        # Constraints
        # Set-covering constraints: Each distribution center must be covered by at least one manufacturing hub.
        for i in range(num_distribution_centers):
            model.addCons(quicksum(transportation[i, j] for j in range(num_manufacturing_hubs)) >= 1, name=f"distribution_coverage_{i}")

        # Logical constraints: A distribution center can only be supplied if the manufacturing hub is operational
        for j in range(num_manufacturing_hubs):
            for i in range(num_distribution_centers):
                model.addCons(transportation[i, j] <= manufacturing_hub[j], name=f"manufacturing_hub_connection_{i}_{j}")

        # Capacity constraints: A manufacturing hub can only supply up to its capacity
        for j in range(num_manufacturing_hubs):
            model.addCons(quicksum(transportation[i, j] * distribution_demand[i] for i in range(num_distribution_centers)) <= manufacturing_capacity[j], name=f"manufacturing_capacity_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_manufacturing_hubs': 11,
        'max_manufacturing_hubs': 350,
        'min_distribution_centers': 9,
        'max_distribution_centers': 112,
    }
    
    optimization = SupplyChainNetworkOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")