import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HelicopterDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        hel_deploy_Costs = np.random.randint(self.min_deployment_cost, self.max_deployment_cost + 1,
                                             (self.n_disaster_sites, self.n_bases))
        base_Maintenance_Costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost + 1, self.n_bases)

        return {
            "hel_deploy_Costs": hel_deploy_Costs,
            "base_Maintenance_Costs": base_Maintenance_Costs
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hel_deploy_Costs = instance['hel_deploy_Costs']
        base_Maintenance_Costs = instance['base_Maintenance_Costs']

        model = Model("HelicopterDeployment")

        # Decision variables
        base_Open = {j: model.addVar(vtype="B", name=f"BaseOpen_{j}") for j in range(self.n_bases)}
        hel_disaster_site = {(i, j): model.addVar(vtype="B", name=f"DisasterSite_{i}_Base_{j}")
                             for i in range(self.n_disaster_sites) for j in range(self.n_bases)}

        # Objective: minimize the total cost (maintenance + deployment costs)
        objective_expr = quicksum(base_Maintenance_Costs[j] * base_Open[j] for j in range(self.n_bases)) + \
                         quicksum(hel_deploy_Costs[i, j] * hel_disaster_site[i, j] for i in range(self.n_disaster_sites)
                                                                                     for j in range(self.n_bases))
        model.setObjective(objective_expr, "minimize")

        # Constraints: Each disaster site is served by exactly one base
        for i in range(self.n_disaster_sites):
            model.addCons(quicksum(hel_disaster_site[i, j] for j in range(self.n_bases)) == 1, f"DisasterSite_{i}")

        # Constraints: A disaster site is served by a base only if the base is open
        for i in range(self.n_disaster_sites):
            for j in range(self.n_bases):
                model.addCons(hel_disaster_site[i, j] <= base_Open[j], f"Serve_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_disaster_sites': 150,
        'n_bases': 75,
        'min_deployment_cost': 75,
        'max_deployment_cost': 300,
        'min_maintenance_cost': 30,
        'max_maintenance_cost': 50,
    }

    helicopter_deployment = HelicopterDeployment(parameters, seed=42)
    instance = helicopter_deployment.generate_instance()
    solve_status, solve_time = helicopter_deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")