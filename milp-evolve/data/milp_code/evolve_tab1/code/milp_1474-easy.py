import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class PharmaceuticalDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        weights = np.random.normal(loc=self.weight_mean, scale=self.weight_std, size=self.number_of_items).astype(int)
        profits = weights + np.random.normal(loc=self.profit_mean_shift, scale=self.profit_std, size=self.number_of_items).astype(int)

        weights = np.clip(weights, self.min_range, self.max_range)
        profits = np.clip(profits, self.min_range, self.max_range)

        capacities = np.zeros(self.number_of_vehicles, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_vehicles,
                                            0.6 * weights.sum() // self.number_of_vehicles,
                                            self.number_of_vehicles - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        temperature_bounds = np.random.uniform(self.min_temp, self.max_temp, self.number_of_items)
        reliability_factors = np.random.uniform(self.min_reliability, self.max_reliability, self.number_of_vehicles)
        compliance_penalties = np.random.uniform(self.min_penalty, self.max_penalty, self.number_of_items)

        return {
            'weights': weights,
            'profits': profits,
            'capacities': capacities,
            'temperature_bounds': temperature_bounds,
            'reliability_factors': reliability_factors,
            'compliance_penalties': compliance_penalties
        }
        
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        temperature_bounds = instance['temperature_bounds']
        reliability_factors = instance['reliability_factors']
        compliance_penalties = instance['compliance_penalties']
        
        number_of_items = len(weights)
        number_of_vehicles = len(capacities)
        
        model = Model("PharmaceuticalDistribution")
        var_names = {}
        temp_vars = {}
        z = {}

        # Decision variables: x[i][j] = 1 if item i is placed in vehicle j
        for i in range(number_of_items):
            for j in range(number_of_vehicles):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
                temp_vars[(i, j)] = model.addVar(vtype="C", name=f"temp_{i}_{j}")
            z[i] = model.addVar(vtype="B", name=f"z_{i}")

        # Objective: Maximize total profit considering compliance penalties and reliability
        objective_expr = quicksum((profits[i]) * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_vehicles))
        reliability_cost = quicksum(reliability_factors[j] * temp_vars[(i, j)] for i in range(number_of_items) for j in range(number_of_vehicles))
        compliance_cost = quicksum(compliance_penalties[i] * (1 - z[i]) for i in range(number_of_items))
        objective_expr -= (reliability_cost + compliance_cost)

        # Constraints: Each item can be in at most one vehicle
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_vehicles)) <= z[i],
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each vehicle must not exceed its capacity
        for j in range(number_of_vehicles):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"VehicleCapacity_{j}"
            )

        # Temperature Constraints: Ensure the temperature remains within bounds
        for i in range(number_of_items):
            for j in range(number_of_vehicles):
                model.addCons(
                    temp_vars[(i, j)] <= temperature_bounds[i] + var_names[(i, j)] * (self.max_temp - self.min_temp),
                    f"TempBoundUpper_{i}_{j}"
                )
                model.addCons(
                    temp_vars[(i, j)] >= temperature_bounds[i] - var_names[(i, j)] * (self.max_temp - self.min_temp),
                    f"TempBoundLower_{i}_{j}"
                )
        
        # Reliability Constraints: ensure z[i] logically connects to x[i][j]
        for i in range(number_of_items):
            for j in range(number_of_vehicles):
                model.addCons(var_names[(i, j)] <= z[i], f"BigM_constraint_1_{i}_{j}")
                model.addCons(var_names[(i, j)] >= z[i] - (1 - var_names[(i, j)]), f"BigM_constraint_2_{i}_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 47
    parameters = {
        'number_of_items': 2000,
        'number_of_vehicles': 2,
        'min_range': 2,
        'max_range': 450,
        'weight_mean': 9,
        'weight_std': 90,
        'profit_mean_shift': 45,
        'profit_std': 30,
        'min_temp': 2,
        'max_temp': 8,
        'min_reliability': 0.73,
        'max_reliability': 3.0,
        'min_penalty': 0,
        'max_penalty': 100,
    }

    distribution = PharmaceuticalDistribution(parameters, seed=seed)
    instance = distribution.generate_instance()
    solve_status, solve_time = distribution.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")