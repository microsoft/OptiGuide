import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HealthcareResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_staff_data(self):
        staff = [f"staff_{i}" for i in range(self.num_staff)]
        departments = [f"dept_{j}" for j in range(self.num_departments)]
        shifts = [f"shift_{k}" for k in range(self.num_shifts)]
        
        skill_levels = {staff[i]: np.random.randint(1, self.max_skill_level + 1) for i in range(self.num_staff)}
        hourly_cost = {staff[i]: np.random.uniform(self.min_hourly_cost, self.max_hourly_cost) for i in range(self.num_staff)}
        
        patient_demand = {departments[j]: np.random.randint(self.min_patients, self.max_patients + 1) for j in range(self.num_departments)}
        
        availability = {
            (staff[i], departments[j], shifts[k]): int(np.random.rand() > self.availability_prob) 
            for i in range(self.num_staff) 
            for j in range(self.num_departments)
            for k in range(self.num_shifts)
        }
        
        res = {
            'staff': staff,
            'departments': departments,
            'shifts': shifts,
            'skill_levels': skill_levels,
            'hourly_cost': hourly_cost,
            'availability': availability,
            'patient_demand': patient_demand
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        staff = instance['staff']
        departments = instance['departments']
        shifts = instance['shifts']
        skill_levels = instance['skill_levels']
        hourly_cost = instance['hourly_cost']
        availability = instance['availability']
        patient_demand = instance['patient_demand']

        model = Model("HealthcareResourceAllocation")

        # Variables
        x = {}
        for i in staff:
            for j in departments:
                for k in shifts:
                    if availability[(i, j, k)]:
                        x[i, j, k] = model.addVar(vtype="B", name=f"x_{i}_{j}_{k}")
        
        # Objective: Minimize total operational cost
        total_cost = quicksum(x[i, j, k] * hourly_cost[i] for i in staff for j in departments for k in shifts if (i, j, k) in x)
        model.setObjective(total_cost, "minimize")
        
        # Constraints

        # Each department's patient demand should be met
        for j in departments:
            total_skilled_hours = quicksum(x[i, j, k] * skill_levels[i] for i in staff for k in shifts if (i, j, k) in x)
            model.addCons(total_skilled_hours >= patient_demand[j], name=f"demand_{j}")
        
        # A staff member can work at most one shift per day
        for i in staff:
            for k in shifts:
                total_shifts = quicksum(x[i, j, k] for j in departments if (i, j, k) in x)
                model.addCons(total_shifts <= 1, name=f"shift_{i}_{k}")

        # A staff member can work at most max_hours per week
        max_weekly_hours = self.max_hours_per_week
        for i in staff:
            total_hours = quicksum(x[i, j, k] for j in departments for k in shifts if (i, j, k) in x)
            model.addCons(total_hours <= max_weekly_hours, name=f"hours_{i}")

        # Total cost cannot exceed the budget
        model.addCons(quicksum(x[i, j, k] * hourly_cost[i] for i in staff for j in departments for k in shifts if (i, j, k) in x) <= self.budget, name="budget")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_staff': 240,
        'num_departments': 10,
        'num_shifts': 28,
        'max_skill_level': 45,
        'min_hourly_cost': 150,
        'max_hourly_cost': 1050,
        'min_patients': 30,
        'max_patients': 450,
        'availability_prob': 0.8,
        'max_hours_per_week': 240,
        'budget': 10000,
    }

    healthcare = HealthcareResourceAllocation(parameters, seed=seed)
    instance = healthcare.generate_staff_data()
    solve_status, solve_time = healthcare.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")