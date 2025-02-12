import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DietPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_meals = self.n_meals
        n_nutrients = self.n_nutrients

        # Random caloric content for each meal
        calorie_content = np.random.randint(200, 800, size=n_meals)

        # Nutritional content matrix (meals x nutrients)
        nutrition_matrix = np.random.randint(0, 20, size=(n_meals, n_nutrients))

        # Daily nutritional requirements
        daily_requirements = np.random.randint(50, 200, size=n_nutrients)

        res = {
            'calorie_content': calorie_content,
            'nutrition_matrix': nutrition_matrix,
            'daily_requirements': daily_requirements
        }

        ### given instance data code ends here
        ### new instance data code ends here

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        calorie_content = instance['calorie_content']
        nutrition_matrix = instance['nutrition_matrix']
        daily_requirements = instance['daily_requirements']
        n_meals = self.n_meals
        n_nutrients = self.n_nutrients
        max_deviation = self.max_deviation

        model = Model("DietPlanning")
        var_names = {}

        # Create variables for each meal
        for m in range(n_meals):
            var_names[m] = model.addVar(vtype="B", name=f"Meal_{m}")

        # Add nutritional constraints to ensure daily requirements are met
        for n in range(n_nutrients):
            model.addCons(
                quicksum(var_names[m] * nutrition_matrix[m, n] for m in range(n_meals)) >= daily_requirements[n],
                f"NutritionalConstraints_{n}"
            )

        # Objective: Minimize deviation from the targeted caloric intake
        target_calories = self.target_calories
        total_calories = quicksum(var_names[m] * calorie_content[m] for m in range(n_meals))
        deviation = model.addVar(vtype="C", name="Deviation")

        model.addCons(total_calories - target_calories <= deviation, name="CaloricDeviationUpper")
        model.addCons(target_calories - total_calories <= deviation, name="CaloricDeviationLower")

        model.setObjective(deviation, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_meals': 100,
        'n_nutrients': 500,
        'max_deviation': 700,
        'target_calories': 2000,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    diet_problem = DietPlanning(parameters, seed=seed)
    instance = diet_problem.generate_instance()
    solve_status, solve_time = diet_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")