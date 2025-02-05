import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

class FactoryProduction:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.Number_of_Tasks > 0 and self.Number_of_Workers > 0
        assert self.Min_Task_Profit >= 0 and self.Max_Task_Profit >= self.Min_Task_Profit
        assert self.Min_Worker_Efficiency >= 0 and self.Max_Worker_Efficiency >= self.Min_Worker_Efficiency
        assert self.Min_Shift_Length > 0 and self.Max_Shift_Length >= self.Min_Shift_Length

        task_profits = np.random.randint(self.Min_Task_Profit, self.Max_Task_Profit + 1, self.Number_of_Tasks)
        worker_efficiencies = np.random.randint(self.Min_Worker_Efficiency, self.Max_Worker_Efficiency + 1, (self.Number_of_Workers, self.Number_of_Tasks))
        shift_lengths = np.random.randint(self.Min_Shift_Length, self.Max_Shift_Length + 1, self.Number_of_Workers)
        
        return {
            "task_profits": task_profits,
            "worker_efficiencies": worker_efficiencies,
            "shift_lengths": shift_lengths,
        }
        
    def solve(self, instance):
        task_profits = instance['task_profits']
        worker_efficiencies = instance['worker_efficiencies']
        shift_lengths = instance['shift_lengths']
        
        model = Model("FactoryWorkerAssignment")
        number_of_workers = len(worker_efficiencies)
        number_of_tasks = len(task_profits)

        # Decision variables
        assignment_vars = {(w, t): model.addVar(vtype="B", name=f"Worker_{w}_Task_{t}") for w in range(number_of_workers) for t in range(number_of_tasks)}

        # Objective: maximize total profit considering worker efficiencies
        model.setObjective(
            quicksum(task_profits[t] * worker_efficiencies[w, t] * assignment_vars[w, t] for w in range(number_of_workers) for t in range(number_of_tasks)), "maximize"
        )
        
        # Constraints: Each task is assigned to at least one worker
        for t in range(number_of_tasks):
            model.addCons(quicksum(assignment_vars[w, t] for w in range(number_of_workers)) >= 1, f"Task_{t}_Assignment")
        
        # Constraints: Workers cannot exceed their shift length
        for w in range(number_of_workers):
            model.addCons(quicksum(worker_efficiencies[w, t] * assignment_vars[w, t] for t in range(number_of_tasks)) <= shift_lengths[w], f"Worker_{w}_Shift")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Tasks': 50,
        'Number_of_Workers': 150,
        'Min_Task_Profit': 3000,
        'Max_Task_Profit': 3000,
        'Min_Worker_Efficiency': 10,
        'Max_Worker_Efficiency': 100,
        'Min_Shift_Length': 250,
        'Max_Shift_Length': 1750,
    }

    factory_worker_optimizer = FactoryProduction(parameters, seed=42)
    instance = factory_worker_optimizer.generate_instance()
    solve_status, solve_time, objective_value = factory_worker_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")