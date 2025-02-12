import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FactoryProjectManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def project_durations(self):
        base_duration = 5.0  # base duration in days
        return base_duration * np.random.rand(self.n_projects, self.n_lines)

    def generate_instance(self):
        project_importance = self.randint(self.n_projects, self.importance_interval)
        line_capacities = self.randint(self.n_lines, self.capacity_interval)
        initialization_costs = self.randint(self.n_lines, self.initialization_cost_interval)
        project_durations = self.project_durations()

        line_capacities = line_capacities * self.ratio * np.sum(project_importance) / np.sum(line_capacities)
        line_capacities = np.round(line_capacities)
        
        break_durations = self.randint(self.n_lines, self.break_duration_interval)

        # New data for worker assignments and auxiliary machines
        worker_assignments = self.randint(self.n_workers, self.worker_assignment_interval)
        auxiliary_machine_capacity = self.randint(self.n_aux_machines, self.aux_machine_capacity_interval)
        
        res = {
            'project_importance': project_importance,
            'line_capacities': line_capacities,
            'initialization_costs': initialization_costs,
            'project_durations': project_durations,
            'break_durations': break_durations,
            'worker_assignments': worker_assignments,
            'auxiliary_machine_capacity': auxiliary_machine_capacity,
        }
        
        return res

    def solve(self, instance):
        # Instance data
        project_importance = instance['project_importance']
        line_capacities = instance['line_capacities']
        initialization_costs = instance['initialization_costs']
        project_durations = instance['project_durations']
        break_durations = instance['break_durations']
        worker_assignments = instance['worker_assignments']
        auxiliary_machine_capacity = instance['auxiliary_machine_capacity']

        n_projects = len(project_importance)
        n_lines = len(line_capacities)

        model = Model("FactoryProjectManagement")

        # Decision variables
        choose_line = {m: model.addVar(vtype="B", name=f"ChooseLine_{m}") for m in range(n_lines)}
        assign_project = {(p, m): model.addVar(vtype="B", name=f"AssignProject_{p}_{m}") for p in range(n_projects) for m in range(n_lines)}
        assign_worker = {(p, w): model.addVar(vtype="B", name=f"AssignWorker_{p}_{w}") for p in range(n_projects) for w in range(self.n_workers)}
        use_aux_machine = {m: model.addVar(vtype="B", name=f"UseAuxMachine_{m}") for m in range(self.n_aux_machines)}

        # Objective: Minimize the total cost including initialization costs, durations penalties, and auxiliary machine costs
        penalty_per_duration = 150
        auxiliary_cost_factor = 500

        objective_expr = quicksum(initialization_costs[m] * choose_line[m] for m in range(n_lines)) + \
                         penalty_per_duration * quicksum(project_durations[p, m] * assign_project[p, m] for p in range(n_projects) for m in range(n_lines)) + \
                         auxiliary_cost_factor * quicksum(auxiliary_machine_capacity[m] * use_aux_machine[m] for m in range(self.n_aux_machines))

        # Constraints: each project must be assigned to exactly one line
        for p in range(n_projects):
            model.addCons(quicksum(assign_project[p, m] for m in range(n_lines)) == 1, f"Project_Assignment_{p}")

        # Constraints: line capacities must be respected
        for m in range(n_lines):
            model.addCons(quicksum(project_importance[p] * assign_project[p, m] for p in range(n_projects)) <= line_capacities[m] * choose_line[m], f"Line_Capacity_{m}")

        # Constraint: Project durations minimized (All projects must fit within the lines' available schedule)
        for p in range(n_projects):
            for m in range(n_lines):
                model.addCons(project_durations[p, m] * assign_project[p, m] <= choose_line[m] * 40, f"Project_Duration_Limit_{p}_{m}")

        # Worker assignment constraints
        for p in range(n_projects):
            model.addCons(quicksum(assign_worker[p, w] for w in range(self.n_workers)) == 1, f"Worker_Assignment_{p}")

        # Auxiliary machine usage constraints
        for m in range(self.n_aux_machines):
            model.addCons(quicksum(project_durations[p, m] * assign_project[p, m] for p in range(n_projects)) <= auxiliary_machine_capacity[m] * use_aux_machine[m], f"Aux_Machine_Usage_{p}_{m}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_projects': 100,
        'n_lines': 50,
        'importance_interval': (50, 500),
        'capacity_interval': (300, 1200),
        'initialization_cost_interval': (200, 800),
        'ratio': 1000.0,
        'break_duration_interval': (30, 90),
        'n_workers': 30,
        'worker_assignment_interval': (1, 10),
        'n_aux_machines': 20,
        'aux_machine_capacity_interval': (10, 50),
    }

    factory_project_management = FactoryProjectManagement(parameters, seed=seed)
    instance = factory_project_management.generate_instance()
    solve_status, solve_time = factory_project_management.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")