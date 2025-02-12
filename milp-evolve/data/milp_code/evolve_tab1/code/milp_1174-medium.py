import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class TaskAssignmentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_employees > 0 and self.n_tasks > 0
        assert self.min_cost_employee >= 0 and self.max_cost_employee >= self.min_cost_employee
        assert self.min_task_duration > 0 and self.max_task_duration >= self.min_task_duration
        assert self.min_task_delay_penalty >= 0 and self.max_task_delay_penalty >= self.min_task_delay_penalty
        assert self.min_overtime_cost >= 0 and self.max_overtime_cost >= self.min_overtime_cost

        operational_costs = np.random.randint(self.min_cost_employee, self.max_cost_employee + 1, self.n_employees)
        task_durations = np.random.randint(self.min_task_duration, self.max_task_duration + 1, self.n_tasks)
        task_dependencies = np.random.randint(0, 2, (self.n_tasks, self.n_tasks))
        delay_penalties = np.random.randint(self.min_task_delay_penalty, self.max_task_delay_penalty + 1, self.n_tasks)
        overtime_costs = np.random.randint(self.min_overtime_cost, self.max_overtime_cost + 1, self.n_employees)
        skill_matrix = np.random.randint(0, 2, (self.n_employees, self.n_tasks))

        return {
            "operational_costs": operational_costs,
            "task_durations": task_durations,
            "task_dependencies": task_dependencies,
            "delay_penalties": delay_penalties,
            "overtime_costs": overtime_costs,
            "skill_matrix": skill_matrix
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        task_durations = instance['task_durations']
        task_dependencies = instance['task_dependencies']
        delay_penalties = instance['delay_penalties']
        overtime_costs = instance['overtime_costs']
        skill_matrix = instance['skill_matrix']

        model = Model("TaskAssignmentOptimization")
        n_employees = len(operational_costs)
        n_tasks = len(task_durations)
        
        employee_vars = {(e, t): model.addVar(vtype="B", name=f"Emp_{e}_Task_{t}") for e in range(n_employees) for t in range(n_tasks)}
        task_completion_vars = {t: model.addVar(vtype="B", name=f"Task_{t}_Completed") for t in range(n_tasks)}
        overtime_vars = {e: model.addVar(vtype="C", name=f"Emp_{e}_Overtime") for e in range(n_employees)}

        # Objective function: Minimize total cost (operational + delay + overtime)
        model.setObjective(
            quicksum(operational_costs[e] * quicksum(employee_vars[e, t] for t in range(n_tasks)) for e in range(n_employees)) +
            quicksum(delay_penalties[t] * (1 - task_completion_vars[t]) for t in range(n_tasks)) +
            quicksum(overtime_costs[e] * overtime_vars[e] for e in range(n_employees)),
            "minimize"
        )

        # Ensure each task is assigned to exactly one employee
        for t in range(n_tasks):
            model.addCons(quicksum(employee_vars[e, t] for e in range(n_employees)) == 1, f"Task_Assignment_{t}")
        
        # Ensure employees can only perform tasks they have skills for
        for e in range(n_employees):
            for t in range(n_tasks):
                model.addCons(employee_vars[e, t] <= skill_matrix[e][t], f"Skill_Constraint_{e}_{t}")
        
        # Task dependency constraints
        for t1 in range(n_tasks):
            for t2 in range(n_tasks):
                if task_dependencies[t1][t2] > 0:
                    model.addCons(task_completion_vars[t1] >= task_completion_vars[t2], f"Task_Dependency_{t1}_to_{t2}")

        # Capacity limits for each employee
        for e in range(n_employees):
            model.addCons(
                quicksum(task_durations[t] * employee_vars[e, t] for t in range(n_tasks)) <= self.max_daily_hours + overtime_vars[e], 
                f"Employee_Capacity_{e}"
            )
        
        # Ensure overtime is non-negative
        for e in range(n_employees):
            model.addCons(overtime_vars[e] >= 0, f"Overtime_NonNegative_{e}")

        # Task completion if assigned
        for t in range(n_tasks):
            model.addCons(task_completion_vars[t] <= quicksum(employee_vars[e, t] for e in range(n_employees)), f"Task_Completed_{t}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_employees': 100,
        'n_tasks': 250,
        'min_cost_employee': 700,
        'max_cost_employee': 2700,
        'min_task_duration': 20,
        'max_task_duration': 32,
        'min_task_delay_penalty': 50,
        'max_task_delay_penalty': 2000,
        'min_overtime_cost': 60,
        'max_overtime_cost': 800,
        'max_daily_hours': 40,
    }
    
    task_optimizer = TaskAssignmentOptimization(parameters, seed)
    instance = task_optimizer.generate_instance()
    solve_status, solve_time, objective_value = task_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")