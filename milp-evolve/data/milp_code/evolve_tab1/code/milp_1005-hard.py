import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class OptimalSchoolDistricting:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def distances(self):
        base_distance = 3.0  # base travel distance in miles
        return base_distance * np.random.rand(self.n_students, self.n_schools)
    
    def performance_levels(self):
        return np.random.randint(self.performance_level_interval[0], self.performance_level_interval[1], self.n_students)

    def generate_instance(self):
        student_performance = self.performance_levels()
        school_capacities = self.randint(self.n_schools, self.capacity_interval)
        activation_costs = self.randint(self.n_schools, self.activation_cost_interval)
        distances = self.distances()

        res = {
            'student_performance': student_performance,
            'school_capacities': school_capacities,
            'activation_costs': activation_costs,
            'distances': distances,
        }

        return res

    def solve(self, instance):
        # Instance data
        student_performance = instance['student_performance']
        school_capacities = instance['school_capacities']
        activation_costs = instance['activation_costs']
        distances = instance['distances']

        n_students = len(student_performance)
        n_schools = len(school_capacities)

        model = Model("OptimalSchoolDistricting")

        # Decision variables
        activate_school = {m: model.addVar(vtype="B", name=f"Activate_School_{m}") for m in range(n_schools)}
        allocate_student = {(n, m): model.addVar(vtype="B", name=f"Allocate_Student_{n}_School_{m}") for n in range(n_students) for m in range(n_schools)}

        # Objective: Minimize total travel distance and balance education equity
        distance_weight = 1.0
        balance_weight = 1.0

        objective_expr = quicksum(distance_weight * distances[n, m] * allocate_student[n, m] for n in range(n_students) for m in range(n_schools)) + \
                         quicksum(balance_weight * student_performance[n] * allocate_student[n, m] for n in range(n_students) for m in range(n_schools))

        # Constraints: each student must be allocated to exactly one school
        for n in range(n_students):
            model.addCons(quicksum(allocate_student[n, m] for m in range(n_schools)) == 1, f"Allocate_Student_{n}")

        # Constraints: school capacity limits must be respected
        for m in range(n_schools):
            model.addCons(quicksum(allocate_student[n, m] for n in range(n_students)) <= school_capacities[m] * activate_school[m], f"School_Capacity_{m}")

        # Constraints: Fairness - balance student performance distribution across schools
        avg_performance = np.mean(student_performance)
        fairness_limit = 0.1 * avg_performance  # Arbitrary fairness limit
        for m in range(n_schools):
            model.addCons(quicksum(allocate_student[n, m] * (student_performance[n] - avg_performance) for n in range(n_students)) <= fairness_limit, f"Ensure_Fairness_{m}")

        # Constraints: Travel distance must be within a feasible limit
        max_distance = 5  # Maximum allowable distance in miles
        for n in range(n_students):
            for m in range(n_schools):
                model.addCons(distances[n, m] * allocate_student[n, m] <= max_distance, f"Travel_Limit_{n}_{m}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_students': 500,
        'n_schools': 40,
        'performance_level_interval': (450, 900),
        'capacity_interval': (40, 80),
        'activation_cost_interval': (1000, 3000),
    }
    
    optimal_school_districting = OptimalSchoolDistricting(parameters, seed=seed)
    instance = optimal_school_districting.generate_instance()
    solve_status, solve_time = optimal_school_districting.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")