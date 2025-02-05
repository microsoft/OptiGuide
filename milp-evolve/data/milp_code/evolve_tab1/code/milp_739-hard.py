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

        # Cost of each meal
        meal_costs = np.random.uniform(1.0, 15.0, size=n_meals)

        # Daily intake limits for fat, protein, and carbs
        fat_limits = np.random.randint(50, 100, size=3)
        protein_limits = np.random.randint(50, 100, size=3)
        carb_limits = np.random.randint(50, 150, size=3)

        res = {
            'calorie_content': calorie_content,
            'nutrition_matrix': nutrition_matrix,
            'daily_requirements': daily_requirements,
            'meal_costs': meal_costs,
            'fat_limits': fat_limits,
            'protein_limits': protein_limits,
            'carb_limits': carb_limits,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        calorie_content = instance['calorie_content']
        nutrition_matrix = instance['nutrition_matrix']
        daily_requirements = instance['daily_requirements']
        meal_costs = instance['meal_costs']
        fat_limits = instance['fat_limits']
        protein_limits = instance['protein_limits']
        carb_limits = instance['carb_limits']
        n_meals = self.n_meals
        n_nutrients = self.n_nutrients
        target_calories = self.target_calories

        model = Model("DietPlanning")
        var_meals = {}
        var_fat_intake = model.addVar(vtype="C", name="FatIntake")
        var_protein_intake = model.addVar(vtype="C", name="ProteinIntake")
        var_carb_intake = model.addVar(vtype="C", name="CarbIntake")

        # Create binary variables for each meal
        for m in range(n_meals):
            var_meals[m] = model.addVar(vtype="B", name=f"Meal_{m}")

        # Add nutritional constraints to ensure daily requirements are met
        for n in range(n_nutrients):
            model.addCons(
                quicksum(var_meals[m] * nutrition_matrix[m, n] for m in range(n_meals)) >= daily_requirements[n],
                f"NutritionalConstraints_{n}"
            )

        # Fat, Protein, Carb intake constraints
        model.addCons(var_fat_intake <= fat_limits[0], "FatIntakeLimit")
        model.addCons(var_protein_intake <= protein_limits[0], "ProteinIntakeLimit")
        model.addCons(var_carb_intake <= carb_limits[0], "CarbIntakeLimit")

        # Objective: Minimize cost and deviation from target calories
        total_calories = quicksum(var_meals[m] * calorie_content[m] for m in range(n_meals))
        total_cost = quicksum(var_meals[m] * meal_costs[m] for m in range(n_meals))
        deviation = model.addVar(vtype="C", name="Deviation")

        model.addCons(total_calories - target_calories <= deviation, name="CaloricDeviationUpper")
        model.addCons(target_calories - total_calories <= deviation, name="CaloricDeviationLower")

        model.setObjective(total_cost + deviation, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_meals': 100,
        'n_nutrients': 2500,
        'target_calories': 1500,
    }

    diet_problem = DietPlanning(parameters, seed=seed)
    instance = diet_problem.generate_instance()
    solve_status, solve_time = diet_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")