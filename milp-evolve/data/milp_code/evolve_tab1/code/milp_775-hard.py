import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DietPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data Generation
    def generate_instance(self):
        n_meals = self.n_meals
        n_nutrients = self.n_nutrients
        n_scenarios = self.n_scenarios

        # Random caloric content for each meal
        calorie_content = np.random.randint(200, 800, size=n_meals)

        # Nutritional content matrix (meals x nutrients)
        nutrition_matrix = np.random.randint(0, 20, size=(n_meals, n_nutrients))

        # Daily nutritional requirements
        daily_requirements = np.random.randint(50, 200, size=n_nutrients)

        # Cost of each meal
        meal_costs = np.random.uniform(1.0, 15.0, size=n_meals)

        # Scenario-based nutritional content variations (scenarios x meals x nutrients)
        nutrition_matrix_scenarios = np.random.normal(loc=nutrition_matrix, scale=self.nutrition_variation, size=(n_scenarios, n_meals, n_nutrients))

        # Scenario-based meal cost variations
        meal_costs_scenarios = np.random.normal(loc=meal_costs, scale=self.cost_variation, size=(n_scenarios, n_meals))

        res = {
            'calorie_content': calorie_content,
            'nutrition_matrix': nutrition_matrix,
            'daily_requirements': daily_requirements,
            'meal_costs': meal_costs,
            'nutrition_matrix_scenarios': nutrition_matrix_scenarios,
            'meal_costs_scenarios': meal_costs_scenarios,
            'n_scenarios': n_scenarios
        }

        return res

    # PySCIPOpt Modeling
    def solve(self, instance):
        calorie_content = instance['calorie_content']
        nutrition_matrix = instance['nutrition_matrix']
        daily_requirements = instance['daily_requirements']
        meal_costs = instance['meal_costs']
        nutrition_matrix_scenarios = instance['nutrition_matrix_scenarios']
        meal_costs_scenarios = instance['meal_costs_scenarios']
        n_meals = self.n_meals
        n_nutrients = self.n_nutrients
        n_scenarios = self.n_scenarios

        model = Model("DietPlanning")
        var_meals = {}

        # Create binary variables for each meal
        for m in range(n_meals):
            var_meals[m] = model.addVar(vtype="B", name=f"Meal_{m}")

        # Add nutritional constraints to ensure daily requirements are met under each scenario
        for n in range(n_nutrients):
            for s in range(n_scenarios):
                model.addCons(
                    quicksum(var_meals[m] * nutrition_matrix_scenarios[s, m, n] for m in range(n_meals)) >= daily_requirements[n],
                    f"NutritionalConstraints_{n}_Scenario_{s}"
                )

        # Objective: Minimize expected cost of selected meals over all scenarios
        expected_cost = quicksum(var_meals[m] * meal_costs_scenarios[s, m] for m in range(n_meals) for s in range(n_scenarios)) / n_scenarios

        model.setObjective(expected_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_meals': 60,
        'n_nutrients': 375,
        'n_scenarios': 50,
        'nutrition_variation': 1.5,
        'cost_variation': 0.25,
    }

    diet_problem = DietPlanning(parameters, seed=seed)
    instance = diet_problem.generate_instance()
    solve_status, solve_time = diet_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")