import random
import time
import numpy as np
from pyscipopt import Model, quicksum


class JobScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def create_random_precedence_constraints(self, n_jobs, n_groups=20):
        # Determine the number of groups and the size of each group
        group_size = n_jobs // n_groups

        # Randomly assign jobs to groups
        jobs = list(range(n_jobs))
        random.shuffle(jobs)
        job_groups = [sorted(jobs[i * group_size:(i + 1) * group_size]) for i in range(n_groups)]

        constraints = []
        
        # Create precedence constraints within each group
        for group in job_groups:
            for i in range(len(group) - 1):
                constraints.append((group[i], group[i + 1]))

        return constraints


    def generate_instance(self):
        processing_times = [random.randint(1, 50) for _ in range(self.n_jobs)] 
        precedence_constraints = self.create_random_precedence_constraints(self.n_jobs, self.n_groups)
        machine_assignment = [random.randint(0, self.n_machines - 1) for _ in range(self.n_jobs)]

        res = {
            'processing_times': processing_times,
            'precedence_constraints': precedence_constraints,
            'machine_assignment': machine_assignment
        }
        return res

    def solve(self, instance):
        processing_times = instance['processing_times']
        precedence_constraints = instance['precedence_constraints']
        machine_assignment = instance['machine_assignment']

        model = Model("JobShopScheduling")
        
        c = {}  # Completion time variables for each job
        s = {}  # Start time variables for each job on its assigned machine
        y = {}  # Binary variables: y[i,k] = 1 if job i starts before job k on the same machine
        end_time = model.addVar("end_time", vtype="C", lb=0)

        for i in range(self.n_jobs):
            c[i] = model.addVar(f"c{i}", vtype="C", lb=0)
            s[i] = model.addVar(f"s{i}", vtype="C", lb=0)

        for i in range(self.n_jobs):
            for k in range(self.n_jobs):
                if i != k:
                    y[i, k] = model.addVar(f"y_{i}_{k}", vtype="B")  # Binary variable: 1 if job i starts before job k


        # Objective: Minimize the makespan (end_time)
        model.setObjective(end_time, "minimize")

        # Job completion and start time constraints
        for i in range(self.n_jobs):
            # Job i's completion time must be its start time + processing time on the assigned machine
            model.addCons(c[i] >= s[i] + processing_times[i], f"time_{i}")

        # Precedence constraints
        for job_before, job_after in precedence_constraints:
            model.addCons(s[job_after] >= c[job_before], f"prec_{job_before}_to_{job_after}")

        # Makespan constraints
        for i in range(self.n_jobs):
            model.addCons(end_time >= c[i], f"makespan_{i}")

        # Sequencing constraints on the same machine
        M = 1e6  # Large constant
        for j in range(self.n_machines):
            jobs_on_machine = [i for i in range(self.n_jobs) if machine_assignment[i] == j]
            for i in range(len(jobs_on_machine)):
                for k in range(i + 1, len(jobs_on_machine)):
                    job_i = jobs_on_machine[i]
                    job_k = jobs_on_machine[k]
                    # Ensure that if job_i and job_k are on the same machine, one must finish before the other starts
                    model.addCons(s[job_k] >= c[job_i] - M * (1 - y[job_i, job_k]), f"seq_{job_i}_{job_k}_machine{j}")
                    model.addCons(s[job_i] >= c[job_k] - M * y[job_i, job_k], f"seq_{job_k}_{job_i}_machine{j}")
                    # Ensure that y[i,k] + y[k,i] = 1 (i.e., either job_i is before job_k or vice versa)
                    model.addCons(y[job_i, job_k] + y[job_k, job_i] == 1, f"order_{job_i}_{job_k}")

        objective_expr = end_time
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_jobs': 150,
        'n_machines': 20,
        'n_groups': 10
    }

    job_scheduling = JobScheduling(parameters)
    instance = job_scheduling.generate_instance()
    solve_status, solve_time = job_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")

