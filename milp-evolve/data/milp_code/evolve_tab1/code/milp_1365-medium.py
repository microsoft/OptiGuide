import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class AdvancedFacilityLocationSustainable:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def diverse_fixed_costs(self, scale=1000):
        return np.random.gamma(shape=2, scale=scale, size=self.n_facilities)

    def transportation_costs_variation(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.abs(
            (rand(self.n_customers, 1) - rand(1, self.n_facilities)) +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.diverse_fixed_costs()
        transportation_costs = self.transportation_costs_variation() * demands[:, np.newaxis]
        priority_customers = self.randint(self.n_customers, (1, 5))  # Priority for each customer

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)

        fish_limits = np.random.uniform(self.min_fish_limit, self.max_fish_limit, self.n_facilities)
        env_impact_limits = np.random.uniform(self.min_env_impact_limit, self.max_env_impact_limit, self.n_facilities)

        penalties = np.random.randint(self.min_penalty, self.max_penalty, self.n_customers)

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'priority_customers': priority_customers,
            'fish_limits': fish_limits,
            'env_impact_limits': env_impact_limits,
            'penalties': penalties
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        priority_customers = instance['priority_customers']
        fish_limits = instance['fish_limits']
        env_impact_limits = instance['env_impact_limits']
        penalties = instance['penalties']

        n_customers = len(demands)
        n_facilities = len(capacities)

        model = Model("AdvancedFacilityLocationSustainable")

        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        unmet_demand = {i: model.addVar(vtype="C", name=f"UnmetDemand_{i}") for i in range(n_customers)}
        env_impact = {j: model.addVar(vtype="C", name=f"EnvImpact_{j}") for j in range(n_facilities)}
        quota_violation = {j: model.addVar(vtype="C", name=f"QuotaViolation_{j}") for j in range(n_facilities)}

        # Objective: minimize the total cost with sustainability and environmental costs
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + \
                         quicksum(self.penalty_unmet * unmet_demand[i] * priority_customers[i] for i in range(n_customers)) + \
                         quicksum(quota_violation[j] * penalties[j] for j in range(n_facilities))

        # Constraints: demand must be met with penalty for unmet demand
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) + unmet_demand[i] >= 1, f"Demand_{i}")

        # Constraints: capacity limits with fluctuations
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * (1 + np.random.uniform(-0.1, 0.1)) * open_facilities[j], f"Capacity_{j}")

        # Environmental impact constraints
        for j in range(n_facilities):
            model.addCons(env_impact[j] == quicksum(serve[i, j] * transportation_costs[i, j] for i in range(n_customers)), f"EnvImpact_{j}")
            model.addCons(env_impact[j] <= env_impact_limits[j], f"EnvImpactLimit_{j}")

        # Fish stock control constraints
        for j in range(n_facilities):
            model.addCons(open_facilities[j] * capacities[j] <= fish_limits[j], f"FishLimit_{j}")

        # Quota Violation Constraints
        for j in range(n_facilities):
            model.addCons(quota_violation[j] >= quicksum(serve[i, j] for i in range(n_customers)) - capacities[j], f"QuotaViolation_{j}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 750,
        'n_facilities': 10,
        'demand_interval': (1, 13),
        'capacity_interval': (90, 1449),
        'ratio': 1.88,
        'penalty_unmet': 200,
        'min_fish_limit': 937,
        'max_fish_limit': 4000,
        'min_env_impact_limit': 150,
        'max_env_impact_limit': 2800,
        'min_penalty': 0,
        'max_penalty': 150,
    }

    facility_location = AdvancedFacilityLocationSustainable(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")