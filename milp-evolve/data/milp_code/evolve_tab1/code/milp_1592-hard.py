import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EmployeeAssignment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        skill_levels = np.random.normal(
            loc=self.skill_mean, 
            scale=self.skill_std, 
            size=(self.number_of_employees, self.number_of_projects)
        ).astype(int)
        
        costs = np.random.normal(
            loc=self.cost_mean, 
            scale=self.cost_std, 
            size=(self.number_of_employees, self.number_of_projects)
        ).astype(int)
        
        hours = np.random.normal(
            loc=self.hours_mean, 
            scale=self.hours_std, 
            size=self.number_of_employees
        ).astype(int)

        budgets = np.random.randint(
            0.4 * costs.sum() / self.number_of_projects, 
            0.6 * costs.sum() / self.number_of_projects, 
            size=self.number_of_projects
        )

        # Ensure non-negative values
        skill_levels = np.clip(skill_levels, self.min_skill, self.max_skill)
        costs = np.clip(costs, self.min_cost, self.max_cost)
        hours = np.clip(hours, self.min_hours, self.max_hours)
        
        res = {
            'skill_levels': skill_levels,
            'costs': costs,
            'hours': hours,
            'budgets': budgets
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        skill_levels = instance['skill_levels']
        costs = instance['costs']
        hours = instance['hours']
        budgets = instance['budgets']
        
        number_of_employees = len(hours)
        number_of_projects = len(budgets)
        
        model = Model("EmployeeAssignment")
        EmployeeVars = {}

        # Decision variables: x[i][j] = 1 if employee i is assigned to project j
        for i in range(number_of_employees):
            for j in range(number_of_projects):
                EmployeeVars[(i, j)] = model.addVar(vtype="B", name=f"EmployeeVars_{i}_{j}")

        # Objective: Maximize total skill coverage
        objective_expr = quicksum(skill_levels[i][j] * EmployeeVars[(i, j)] for i in range(number_of_employees) for j in range(number_of_projects))

        # Constraints: Each employee can only work on projects up to their available hours
        for i in range(number_of_employees):
            model.addCons(
                quicksum(EmployeeVars[(i, j)] for j in range(number_of_projects)) <= hours[i],
                f"ScheduleConstraints_{i}"
            )

        # Constraints: Total cost for each project must not exceed the project budget
        for j in range(number_of_projects):
            model.addCons(
                quicksum(costs[i][j] * EmployeeVars[(i, j)] for i in range(number_of_employees)) <= budgets[j],
                f"BudgetConstraints_{j}"
            )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_employees': 50,
        'number_of_projects': 80,
        'skill_mean': 60,
        'skill_std': 30,
        'min_skill': 6,
        'max_skill': 2000,
        'cost_mean': 2500,
        'cost_std': 600,
        'min_cost': 100,
        'max_cost': 3000,
        'hours_mean': 200,
        'hours_std': 50,
        'min_hours': 30,
        'max_hours': 350,
    }

    assignment = EmployeeAssignment(parameters, seed=seed)
    instance = assignment.generate_instance()
    solve_status, solve_time = assignment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")