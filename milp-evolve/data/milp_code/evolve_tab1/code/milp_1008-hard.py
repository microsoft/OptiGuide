import random
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class TaskSchedulingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_tasks >= self.n_facilities
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_variable_cost >= 0 and self.max_variable_cost >= self.min_variable_cost
        assert self.min_task_duration > 0 and self.max_task_duration >= self.min_task_duration
        
        fixed_costs = np.random.uniform(self.min_fixed_cost, self.max_fixed_cost, self.n_facilities).tolist()
        variable_costs = np.random.uniform(self.min_variable_cost, self.max_variable_cost, (self.n_facilities, self.n_tasks)).tolist()
        task_durations = np.random.uniform(self.min_task_duration, self.max_task_duration, self.n_tasks).tolist()

        efficiency_scores = np.random.uniform(0.5, 1.5, self.n_facilities).tolist()
        employee_availability = np.random.uniform(0.5, 1.5, self.n_employees).tolist()

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, self.n_facilities - 1)
            fac2 = random.randint(0, self.n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))
        
        return {
            "fixed_costs": fixed_costs,
            "variable_costs": variable_costs,
            "task_durations": task_durations,
            "efficiency_scores": efficiency_scores,
            "employee_availability": employee_availability,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        variable_costs = instance['variable_costs']
        task_durations = instance['task_durations']
        efficiency_scores = instance['efficiency_scores']
        employee_availability = instance['employee_availability']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        
        model = Model("TaskSchedulingOptimization")
        n_facilities = len(fixed_costs)
        n_tasks = len(task_durations)
        n_employees = len(employee_availability)
        
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        task_vars = {(f, t): model.addVar(vtype="B", name=f"Task_{t}_Facility_{f}") for f in range(n_facilities) for t in range(n_tasks)}
        time_used_vars = {f: model.addVar(vtype="C", name=f"TimeUsed_{f}", lb=0) for f in range(n_facilities)}
        operational_time_vars = {f: model.addVar(vtype="C", name=f"OperationalTime_{f}", lb=0) for f in range(n_facilities)}
        employee_effort_vars = {(f, e): model.addVar(vtype="C", name=f"EmployeeEffort_{f}_{e}", lb=0) for f in range(n_facilities) for e in range(n_employees)}

        model.setObjective(
            quicksum(efficiency_scores[f] * task_vars[f, t] for f in range(n_facilities) for t in range(n_tasks)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(variable_costs[f][t] * task_vars[f, t] for f in range(n_facilities) for t in range(n_tasks)),
            "maximize"
        )

        # Each task must be assigned to exactly one facility
        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[f, t] for f in range(n_facilities)) == 1, f"Task_{t}_Assignment")

        # A facility can only handle a task if it is selected
        for f in range(n_facilities):
            for t in range(n_tasks):
                model.addCons(task_vars[f, t] <= facility_vars[f], f"Facility_{f}_Task_{t}")
        
        # Capacity constraint in terms of time used by each facility
        for f in range(n_facilities):
            model.addCons(time_used_vars[f] == quicksum(task_vars[f, t] * task_durations[t] for t in range(n_tasks)), f"TimeUsed_{f}")
        
        # Facility's operational time should not exceed its limit
        for f in range(n_facilities):
            model.addCons(operational_time_vars[f] <= self.max_operational_time, f"OperationalTime_{f}_Limit")

        # Employee effort is limited to availability
        for e in range(n_employees):
            model.addCons(quicksum(employee_effort_vars[f, e] for f in range(n_facilities)) <= employee_availability[e], f"Employee_{e}_EffortLimit")
        
        # Mutual exclusivity constraint
        for i, (fac1, fac2) in enumerate(mutual_exclusivity_pairs):
            model.addCons(facility_vars[fac1] + facility_vars[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        model.optimize()
        
        return model.getStatus(), model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 60,
        'n_tasks': 100,
        'n_employees': 150,
        'min_fixed_cost': 1500,
        'max_fixed_cost': 2000,
        'min_variable_cost': 350,
        'max_variable_cost': 900,
        'min_task_duration': 50,
        'max_task_duration': 80,
        'max_operational_time': 400,
        'n_exclusive_pairs': 40,
    }

    task_optimizer = TaskSchedulingOptimization(parameters, seed)
    instance = task_optimizer.generate_instance()
    status, objective_value = task_optimizer.solve(instance)

    print(f"Solve Status: {status}")
    print(f"Objective Value: {objective_value:.2f}")