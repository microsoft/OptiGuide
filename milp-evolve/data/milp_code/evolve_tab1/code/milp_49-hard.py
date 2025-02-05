import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MealPlanningMILP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_meals(self, n_meals, n_nutrients):
        required_nutrients = np.random.randint(20, 80, size=n_nutrients).tolist()
        meals = {f'm_{m}': np.random.randint(0, 100, size=n_nutrients).tolist() for m in range(n_meals)}
        costs = np.random.randint(5, 15, size=n_meals).tolist()
        return meals, costs, required_nutrients

    def generate_instances(self):
        n_meals = np.random.randint(self.min_n, self.max_n + 1)
        n_nutrients = np.random.randint(self.min_nutr, self.max_nutr)
        
        meals, meal_cost, required_nutrients = self.generate_meals(n_meals, n_nutrients)
        
        res = {
            'meals': meals,
            'meal_cost': meal_cost,
            'required_nutrients': required_nutrients,
            'n_meals': n_meals,
            'n_nutrients': n_nutrients
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        meals = instance['meals']
        meal_cost = instance['meal_cost']
        required_nutrients = instance['required_nutrients']
        n_meals = instance['n_meals']
        n_nutrients = instance['n_nutrients']

        model = Model("Meal_Planning_MILP")
        meal_vars = {}
        nutrient_vars = {}
        
        # Create variables for each meal
        for meal in meals:
            meal_vars[meal] = model.addVar(vtype="B", name=meal)
        
        # Objective function - minimize the cost of procuring meals
        objective_expr = quicksum(
            meal_cost[m] * meal_vars[f'm_{m}'] for m in range(n_meals)
        )
        
        # Add nutrient constraints
        for n in range(n_nutrients):
            model.addCons(
                quicksum(meals[f"m_{m}"][n] * meal_vars[f"m_{m}"] for m in range(n_meals)) >= required_nutrients[n],
                name=f"nutrient_{n}_requirement"
            )

        # Ensure diversity in meal selection
        model.addCons(quicksum(meal_vars[f"m_{m}"] for m in range(n_meals)) <= self.max_meals_per_day, name="meal_selection_limit")
        
        model.setObjective(objective_expr, "minimize")
       
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 100,
        'max_n': 2100,
        'min_nutr': 25,
        'max_nutr': 150,
        'max_meals_per_day': 60,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    meal_planning_milp = MealPlanningMILP(parameters, seed=seed)
    instance = meal_planning_milp.generate_instances()
    solve_status, solve_time = meal_planning_milp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")