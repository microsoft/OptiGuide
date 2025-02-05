import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ManufacturingPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_assembly_lines > 0 and self.n_neighborhoods >= self.n_assembly_lines
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_variable_cost >= 0 and self.max_variable_cost >= self.min_variable_cost
        assert self.min_effort > 0 and self.max_effort >= self.min_effort

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_assembly_lines)
        variable_costs = np.random.randint(self.min_variable_cost, self.max_variable_cost + 1, (self.n_assembly_lines, self.n_neighborhoods))
        efforts = np.random.normal((self.min_effort + self.max_effort) / 2, 
                                       (self.max_effort - self.min_effort) / 10, 
                                       self.n_assembly_lines)
        production_rewards = np.random.uniform(10, 100, self.n_neighborhoods)
        product_demands = np.random.normal(1, 0.1, (self.n_assembly_lines, self.n_neighborhoods))

        return {
            "fixed_costs": fixed_costs,
            "variable_costs": variable_costs,
            "efforts": efforts,
            "production_rewards": production_rewards,
            "product_demands": product_demands
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        variable_costs = instance['variable_costs']
        efforts = instance['efforts']
        production_rewards = instance['production_rewards']
        product_demands = instance['product_demands']

        model = Model("ManufacturingPlanning")
        n_assembly_lines = len(fixed_costs)
        n_neighborhoods = len(variable_costs[0])
        
        line_vars = {l: model.addVar(vtype="B", name=f"Line_{l}") for l in range(n_assembly_lines)}
        effort_vars = {(l, n): model.addVar(vtype="C", name=f"Line_{l}_Effort_{n}") for l in range(n_assembly_lines) for n in range(n_neighborhoods)}

        model.setObjective(
            quicksum(production_rewards[n] * effort_vars[l, n] for l in range(n_assembly_lines) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[l] * line_vars[l] for l in range(n_assembly_lines)) -
            quicksum(variable_costs[l][n] * effort_vars[l, n] for l in range(n_assembly_lines) for n in range(n_neighborhoods)),
            "maximize"
        )

        for n in range(n_neighborhoods):
            model.addCons(quicksum(effort_vars[l, n] for l in range(n_assembly_lines)) >= 1, f"Product_{n}_Production")

        for l in range(n_assembly_lines):
            for n in range(n_neighborhoods):
                model.addCons(effort_vars[l, n] <= line_vars[l], f"Line_{l}_Effort_{n}")

        for l in range(n_assembly_lines):
            model.addCons(quicksum(product_demands[l][n] * effort_vars[l, n] for n in range(n_neighborhoods)) <= efforts[l], f"Line_{l}_Capacity")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_assembly_lines': 90,
        'n_neighborhoods': 240,
        'min_fixed_cost': 300,
        'max_fixed_cost': 360,
        'min_variable_cost': 160,
        'max_variable_cost': 160,
        'min_effort': 30,
        'max_effort': 350,
    }

    manufacturing_planner = ManufacturingPlanning(parameters, seed=seed)
    instance = manufacturing_planner.generate_instance()
    solve_status, solve_time, objective_value = manufacturing_planner.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")