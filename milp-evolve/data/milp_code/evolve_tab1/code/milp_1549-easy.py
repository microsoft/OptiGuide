import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MealSuppliesDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nutrition = np.random.normal(loc=self.nutrition_mean, scale=self.nutrition_std, size=self.number_of_supplies).astype(int)
        costs = nutrition + np.random.normal(loc=self.cost_shift, scale=self.cost_std, size=self.number_of_supplies).astype(int)

        # Ensure non-negative values
        nutrition = np.clip(nutrition, self.min_nutrition, self.max_nutrition)
        costs = np.clip(costs, self.min_cost, self.max_cost)

        holding_capacities = np.random.randint(
            0.4 * costs.sum() // self.number_of_shelters,
            0.6 * costs.sum() // self.number_of_shelters,
            size=self.number_of_shelters
        )

        res = {'nutrition': nutrition,
               'costs': costs,
               'holding_capacities': holding_capacities}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        nutrition = instance['nutrition']
        costs = instance['costs']
        holding_capacities = instance['holding_capacities']
        
        number_of_supplies = len(nutrition)
        number_of_shelters = len(holding_capacities)
        
        model = Model("MealSuppliesDistribution")
        NewVars = {}

        # Decision variables: x[i][j] = 1 if supply unit i is allocated to shelter j
        for i in range(number_of_supplies):
            for j in range(number_of_shelters):
                NewVars[(i, j)] = model.addVar(vtype="B", name=f"NewVars_{i}_{j}")

        # Objective: Maximize total nutrition
        objective_expr = quicksum(nutrition[i] * NewVars[(i, j)] for i in range(number_of_supplies) for j in range(number_of_shelters))

        # Constraints: Each supply can be in at most one shelter
        for i in range(number_of_supplies):
            model.addCons(
                quicksum(NewVars[(i, j)] for j in range(number_of_shelters)) <= 1,
                f"SupplyAssignment_{i}"
            )

        # Constraints: Total cost in each shelter must not exceed its capacity
        for j in range(number_of_shelters):
            model.addCons(
                quicksum(costs[i] * NewVars[(i, j)] for i in range(number_of_supplies)) <= holding_capacities[j],
                f"HoldingCapacity_{j}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_supplies': 3000,
        'number_of_shelters': 2,
        'min_nutrition': 0,
        'max_nutrition': 2250,
        'nutrition_mean': 75,
        'nutrition_std': 300,
        'cost_shift': 3,
        'cost_std': 1,
        'min_cost': 0,
        'max_cost': 187,
    }

    distribution = MealSuppliesDistribution(parameters, seed=seed)
    instance = distribution.generate_instance()
    solve_status, solve_time = distribution.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")