import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class MilitaryAssetDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def uniform(self, size, interval):
        return np.random.uniform(interval[0], interval[1], size)

    def generate_instance(self):
        asset_capacities = self.randint(self.n_assets, self.capacity_interval)
        deployment_costs = self.uniform((self.n_zones, self.n_assets), self.cost_interval)
        risk_levels = self.uniform(self.n_zones, self.risk_interval)
        asset_fixed_costs = self.randint(self.n_assets, self.fixed_cost_interval)

        res = {
            'asset_capacities': asset_capacities,
            'deployment_costs': deployment_costs,
            'risk_levels': risk_levels,
            'asset_fixed_costs': asset_fixed_costs
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        asset_capacities = instance['asset_capacities']
        deployment_costs = instance['deployment_costs']
        risk_levels = instance['risk_levels']
        asset_fixed_costs = instance['asset_fixed_costs']
        
        n_zones = len(risk_levels)
        n_assets = len(asset_capacities)

        model = Model("MilitaryAssetDeployment")

        # Decision variables
        deploy = {
            (i, j): model.addVar(vtype="B", name=f"Deploy_{i}_{j}") 
            for i in range(n_zones) for j in range(n_assets)
        }
        use_asset = {j: model.addVar(vtype="B", name=f"Use_{j}") for j in range(n_assets)}

        # Objective: minimize the total deployment and risk cost
        objective_expr = (
            quicksum(deployment_costs[i, j] * deploy[i, j] for i in range(n_zones) for j in range(n_assets)) +
            quicksum(asset_fixed_costs[j] * use_asset[j] for j in range(n_assets)) +
            quicksum(risk_levels[i] * quicksum(deploy[i, j] for j in range(n_assets)) for i in range(n_zones))
        )

        # Constraints: Ensuring that each zone is covered
        for i in range(n_zones):
            model.addCons(quicksum(deploy[i, j] for j in range(n_assets)) >= 1, f"ZoneCovered_{i}")

        # Constraints: Capacity limits for each asset
        for j in range(n_assets):
            model.addCons(
                quicksum(deploy[i, j] for i in range(n_zones)) <= asset_capacities[j] * use_asset[j], 
                f"Capacity_{j}"
            )
        
        # Constraints: Each asset can only be used if it is deployed
        for i in range(n_zones):
            for j in range(n_assets):
                model.addCons(deploy[i, j] <= use_asset[j], f"AssetDeployed_{i}_{j}")

        ### Additional constraints for risk management ###
        # Constraint: limiting the maximum allowable risk level
        max_risk_limit = self.max_risk_limit
        model.addCons(
            quicksum(risk_levels[i] * quicksum(deploy[i, j] for j in range(n_assets)) for i in range(n_zones)) <= max_risk_limit,
            "MaxRiskLimit"
        )

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 12345
    parameters = {
        'n_zones': 200,
        'n_assets': 50,
        'capacity_interval': (30, 300),
        'cost_interval': (200.0, 2000.0),
        'risk_interval': (0.38, 0.86),
        'fixed_cost_interval': (1000, 5000),
        'max_risk_limit': 800.0,
    }

    deployment = MilitaryAssetDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")