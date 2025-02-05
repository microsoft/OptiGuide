import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MealServiceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_halls > 0 and self.n_meals > 0
        assert self.min_prep_cost >= 0 and self.max_prep_cost >= self.min_prep_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost

        preparation_costs = np.random.randint(self.min_prep_cost, self.max_prep_cost + 1, self.n_meals)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_meals, self.n_halls))

        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_halls).astype(int)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_halls)

        return {
            "preparation_costs": preparation_costs,
            "transport_costs": transport_costs,
            "demands": demands,
            "capacities": capacities
        }

    # MILP modeling
    def solve(self, instance):
        preparation_costs = instance["preparation_costs"]
        transport_costs = instance["transport_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]

        model = Model("MealServiceOptimization")
        n_halls = len(capacities)
        n_meals = len(preparation_costs)

        BigM = self.bigM

        # Decision variables
        meal_vars = {m: model.addVar(vtype="B", name=f"Meal_{m}") for m in range(n_meals)}
        distribution_vars = {(m, h): model.addVar(vtype="B", name=f"Meal_{m}_Hall_{h}") for m in range(n_meals) for h in range(n_halls)}

        # Objective function
        model.setObjective(
            quicksum(preparation_costs[m] * meal_vars[m] for m in range(n_meals)) +
            quicksum(transport_costs[m, h] * distribution_vars[m, h] for m in range(n_meals) for h in range(n_halls)),
            "minimize"
        )

        # Constraints: Each hall's demand should be met by at least one meal
        for h in range(n_halls):
            model.addCons(quicksum(distribution_vars[m, h] for m in range(n_meals)) >= 1, f"Hall_{h}_Demand")

        # Constraints: Halls cannot exceed their capacity
        for h in range(n_halls):
            model.addCons(quicksum(demands[h] * distribution_vars[m, h] for m in range(n_meals)) <= capacities[h], f"Hall_{h}_Capacity")

        # Constraints: Only prepared meals can be distributed
        for m in range(n_meals):
            for h in range(n_halls):
                model.addCons(distribution_vars[m, h] <= meal_vars[m], f"Meal_{m}_Serve_{h}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value

if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_halls': 168,
        'n_meals': 30,
        'min_prep_cost': 1800,
        'max_prep_cost': 3000,
        'min_transport_cost': 10,
        'max_transport_cost': 1000,
        'mean_demand': 3,
        'std_dev_demand': 5,
        'min_capacity': 75,
        'max_capacity': 1250,
        'bigM': 750,
    }

    meal_optimizer = MealServiceOptimization(parameters, seed)
    instance = meal_optimizer.generate_instance()
    solve_status, solve_time, objective_value = meal_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")