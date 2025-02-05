import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class Factory:
    """Helper function: Container for tasks, machines and shifts."""
    def __init__(self, number_of_tasks, number_of_machines, number_of_shifts):
        self.number_of_tasks = number_of_tasks
        self.number_of_machines = number_of_machines
        self.number_of_shifts = number_of_shifts
        self.tasks = np.arange(number_of_tasks)
        self.machines = np.arange(number_of_machines)
        self.shifts = np.arange(number_of_shifts)

class ManufacturingSchedulingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_factory(self):
        return Factory(self.N_tasks, self.N_machines, self.N_shifts)

    def generate_instance(self):
        factory = self.generate_factory()
        labor_costs = np.random.randint(50, 150, size=(self.N_tasks, self.N_shifts))
        energy_costs = np.random.randint(20, 100, size=(self.N_tasks, self.N_machines))
        environment_penalties = np.random.randint(10, 50, size=self.N_machines)
        machine_capacities = np.random.randint(5, 15, size=self.N_machines)
        max_workload = np.random.randint(10, 20, size=self.N_tasks)
        delayed_costs = np.random.randint(100, 300, size=(self.N_tasks, self.N_shifts))

        res = {
            'factory': factory,
            'labor_costs': labor_costs,
            'energy_costs': energy_costs,
            'environment_penalties': environment_penalties,
            'machine_capacities': machine_capacities,
            'max_workload': max_workload,
            'delayed_costs': delayed_costs,
            'N_tasks': self.N_tasks,
            'N_machines': self.N_machines,
            'N_shifts': self.N_shifts,
            'N_time_limit': self.N_time_limit
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        factory = instance['factory']
        labor_costs = instance['labor_costs']
        energy_costs = instance['energy_costs']
        environment_penalties = instance['environment_penalties']
        machine_capacities = instance['machine_capacities']
        max_workload = instance['max_workload']
        delayed_costs = instance['delayed_costs']
        N_time_limit = instance['N_time_limit']

        model = Model("ManufacturingSchedulingOptimization")

        # Variables
        H_worker_shift = {(t, s): model.addVar(vtype="B", name=f"H_worker_shift_{t}_{s}") for t in factory.tasks for s in factory.shifts}
        M_machine_task = {(m, t): model.addVar(vtype="B", name=f"M_machine_task_{m}_{t}") for m in factory.machines for t in factory.tasks}
        L_task_shift = {t: model.addVar(vtype="B", name=f"L_task_shift_{t}") for t in factory.tasks}
        P_task_delay = {(t, s): model.addVar(vtype="B", name=f"P_task_delay_{t}_{s}") for t in factory.tasks for s in factory.shifts}

        # Constraints
        for t in factory.tasks:
            model.addCons(quicksum(H_worker_shift[t, s] for s in factory.shifts) == 1, name=f"TaskShift_{t}")
        
        for s in factory.shifts:
            for m in factory.machines:
                model.addCons(quicksum(M_machine_task[m, t] for t in factory.tasks) <= machine_capacities[m], name=f"MachineCapacity_{m}_{s}")

        for t in factory.tasks:
            for s in factory.shifts:
                model.addCons(quicksum(M_machine_task[m, t] for m in factory.machines) >= H_worker_shift[t, s], name=f"MachineTaskAssignment_{t}_{s}")
                model.addCons(H_worker_shift[t, s] <= L_task_shift[t], name=f"WorkerAvailable_{t}_{s}")

        # New Constraints for Stochastic Delays
        for t in factory.tasks:
            model.addCons(quicksum(P_task_delay[t, s] for s in factory.shifts) <= 1, name=f"TaskDelay_{t}")
            for s in factory.shifts:
                model.addCons(P_task_delay[t, s] + H_worker_shift[t, s] <= 1, name=f"NoConcurrentTaskAndDelay_{t}_{s}")

        labor_cost = quicksum(H_worker_shift[t, s] * labor_costs[t, s] for t in factory.tasks for s in factory.shifts)
        energy_cost = quicksum(M_machine_task[m, t] * energy_costs[t, m] for m in factory.machines for t in factory.tasks)
        environment_penalty = quicksum(M_machine_task[m, t] * environment_penalties[m] for m in factory.machines for t in factory.tasks)
        delay_cost = quicksum(P_task_delay[t, s] * delayed_costs[t, s] for t in factory.tasks for s in factory.shifts)

        # Objective Function
        model.setObjective(labor_cost + energy_cost + environment_penalty + delay_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'N_tasks': 150,
        'N_machines': 20,
        'N_shifts': 30,
        'N_time_limit': 3600,
    }

    manufacturing_scheduling_optimization = ManufacturingSchedulingOptimization(parameters, seed=seed)
    instance = manufacturing_scheduling_optimization.generate_instance()
    solve_status, solve_time = manufacturing_scheduling_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")