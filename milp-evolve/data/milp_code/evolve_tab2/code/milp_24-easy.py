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

    def create_random_precedence_constraints(self, n_jobs, dynamic_groups=None):
        # Determine the number of groups and adaptively change based on job count
        if not dynamic_groups:
            n_groups = np.random.randint(5, 15)
        else:
            n_groups = dynamic_groups
        group_size = max(n_jobs // n_groups, 1)

        # Randomly assign jobs to groups
        jobs = list(range(n_jobs))
        random.shuffle(jobs)
        job_groups = [sorted(jobs[i * group_size:(i + 1) * group_size]) for i in range(min(n_groups, len(jobs) // group_size))]

        constraints = []
        
        # Create precedence constraints within each group
        for group in job_groups:
            for i in range(len(group) - 1):
                constraints.append((group[i], group[i + 1]))

        # Introduce random inter-group precedence constraints
        for _ in range(np.random.randint(1, 10)):
            if len(jobs) > 1:
                job_before = random.choice(jobs)
                job_after = random.choice([j for j in jobs if j != job_before])
                constraints.append((job_before, job_after))

        return constraints

    def generate_instance(self):
        processing_times = [random.randint(1, 50) for _ in range(self.n_jobs)] 
        precedence_constraints = self.create_random_precedence_constraints(self.n_jobs)
        machine_assignment = [random.randint(0, self.n_machines - 1) for _ in range(self.n_jobs)]
        machine_costs = [random.uniform(5, 20) for _ in range(self.n_machines)]

        res = {
            'processing_times': processing_times,
            'precedence_constraints': precedence_constraints,
            'machine_assignment': machine_assignment,
            'machine_costs': machine_costs
        }
        return res

    def solve(self, instance):
        processing_times = instance['processing_times']
        precedence_constraints = instance['precedence_constraints']
        machine_assignment = instance['machine_assignment']
        machine_costs = instance['machine_costs']

        model = Model("JobShopScheduling")
        
        c = {}  
        s = {}  
        y = {}  
        x = {}  

        end_time = model.addVar("end_time", vtype="C", lb=0)
        assignment_cost = model.addVar("assignment_cost", vtype="C", lb=0)

        for i in range(self.n_jobs):
            c[i] = model.addVar(f"c{i}", vtype="C", lb=0)
            s[i] = model.addVar(f"s{i}", vtype="C", lb=0)
            x[i] = model.addVar(f"x{i}", vtype="C", lb=0)  # Cost variable

        for i in range(self.n_jobs):
            for k in range(self.n_jobs):
                if i != k:
                    y[i, k] = model.addVar(f"y_{i}_{k}", vtype="B")

        # Modified Objective: Minimize makespan and machine assignment cost
        model.setObjective(0.5 * end_time + 0.5 * assignment_cost, "minimize")

        # Enforce completion and start time constraints
        for i in range(self.n_jobs):
            model.addCons(c[i] >= s[i] + processing_times[i], f"time_{i}")
            model.addCons(assignment_cost >= x[i] + machine_costs[machine_assignment[i]], f"cost_{i}")

        # Enforce precedence constraints
        for job_before, job_after in precedence_constraints:
            model.addCons(s[job_after] >= c[job_before], f"prec_{job_before}_{job_after}")

        # Enforce makespan constraints
        for i in range(self.n_jobs):
            model.addCons(end_time >= c[i], f"makespan_{i}")

        # Enforce sequencing constraints on the same machine
        M = 1e6
        for j in range(self.n_machines):
            jobs_on_machine = [i for i in range(self.n_jobs) if machine_assignment[i] == j]
            for idx, job_i in enumerate(jobs_on_machine):
                for job_k in jobs_on_machine[idx+1:]:
                    model.addCons(s[job_k] >= c[job_i] - M * (1 - y[job_i, job_k]), f"seq_{job_i}_to_{job_k}_machine{j}")
                    model.addCons(s[job_i] >= c[job_k] - M * y[job_i, job_k], f"seq_{job_k}_to_{job_i}_machine{j}")
                    model.addCons(y[job_i, job_k] + y[job_k, job_i] == 1, f"order_{job_i}_{job_k}")

        start_time = time.time()
        model.optimize()
        solve_end_time = time.time()

        return model.getStatus(), solve_end_time - start_time

if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_jobs': 1200,
        'n_machines': 160,
    }

    job_scheduling = JobScheduling(parameters)
    instance = job_scheduling.generate_instance()
    solve_status, solve_time = job_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")