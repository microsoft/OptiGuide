import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MultiFactoryScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_factories(self):
        factories = {}
        for i in range(self.num_factories):
            capacity = np.random.randint(self.min_capacity, self.max_capacity+1)
            setup_cost = np.random.uniform(*self.setup_cost_range)
            factories[i] = {'capacity': capacity, 'setup_cost': setup_cost}
        return factories

    def generate_tasks(self):
        tasks = []
        for i in range(self.num_tasks):
            processing_time = np.random.uniform(*self.processing_time_range)
            tasks.append({'processing_time': processing_time})
        return tasks

    def generate_instance(self):
        factories = self.generate_factories()
        tasks = self.generate_tasks()
        res = {'factories': factories, 'tasks': tasks}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        factories = instance['factories']
        tasks = instance['tasks']
      
        model = Model("MultiFactoryScheduling")

        # Decision Variables
        T_vars = {
            (i, j): model.addVar(vtype="B", name=f"T_{i}_{j}")
            for i in factories for j in range(self.num_tasks)
        }
        S_vars = {
            i: model.addVar(vtype="B", name=f"S_{i}")
            for i in factories
        }
        P_vars = {
            (i, j): model.addVar(vtype="C", name=f"P_{i}_{j}")
            for i in factories for j in range(self.num_tasks)
        }

        # Objective Function: Minimize total cost
        objective_expr = quicksum(
            tasks[j]['processing_time'] * T_vars[i, j]
            for i in factories for j in range(self.num_tasks)
        ) + quicksum(
            factories[i]['setup_cost'] * S_vars[i]
            for i in factories
        )
        model.setObjective(objective_expr, "minimize")

        # Constraints: Each task must be assigned to exactly one factory
        for j in range(self.num_tasks):
            model.addCons(
                quicksum(T_vars[i, j] for i in factories) == 1,
                f"task_assignment_{j}"
            )

        # Constraints: Factory capacity
        for i in factories:
            model.addCons(
                quicksum(tasks[j]['processing_time'] * T_vars[i, j] for j in range(self.num_tasks)) <= factories[i]['capacity'] * S_vars[i],
                f"capacity_{i}"
            )

        # Constraints: Max parallel tasks
        for i in factories:
            for t in range(self.max_parallel_tasks):
                model.addCons(
                    quicksum(T_vars[i, j] for j in range(t, self.num_tasks, self.max_parallel_tasks)) <= self.num_parallel_limits,
                    f"parallel_limit_{i}_{t}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 24
    parameters = {
        'num_factories': 40,
        'num_tasks': 140,
        'min_capacity': 500,
        'max_capacity': 600,
        'setup_cost_range': (400, 2000),
        'processing_time_range': (9, 90),
        'max_parallel_tasks': 8,
        'num_parallel_limits': 4,
    }

    production_scheduler = MultiFactoryScheduling(parameters, seed=seed)
    instance = production_scheduler.generate_instance()
    solve_status, solve_time = production_scheduler.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")