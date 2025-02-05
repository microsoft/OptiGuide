import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class JobSchedulingRobust:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def create_random_precedence_constraints(self, n_jobs, n_groups=20):
        group_size = n_jobs // n_groups
        jobs = list(range(n_jobs))
        random.shuffle(jobs)
        job_groups = [sorted(jobs[i * group_size:(i + 1) * group_size]) for i in range(n_groups)]
        
        constraints = []
        for group in job_groups:
            for i in range(len(group) - 1):
                constraints.append((group[i], group[i + 1]))
        return constraints

    def generate_instance(self):
        processing_time_means = np.random.uniform(10, 40, size=self.n_jobs)
        processing_time_vars = np.random.uniform(5, 10, size=self.n_jobs)  # Variability in processing times
        processing_times = np.random.normal(processing_time_means, processing_time_vars).clip(min=1)
        
        precedence_constraints = self.create_random_precedence_constraints(self.n_jobs, self.n_groups)
        machine_assignment = [random.randint(0, self.n_machines - 1) for _ in range(self.n_jobs)]
        
        tariff_impacts = np.random.normal(self.tariff_mean, self.tariff_std, size=self.n_jobs)

        res = {
            'processing_times': processing_times,
            'precedence_constraints': precedence_constraints,
            'machine_assignment': machine_assignment,
            'tariff_impacts': tariff_impacts
        }
        return res

    def solve(self, instance):
        processing_times = instance['processing_times']
        precedence_constraints = instance['precedence_constraints']
        machine_assignment = instance['machine_assignment']
        tariff_impacts = instance['tariff_impacts']

        model = Model("RobustJobShopScheduling")
        
        c = {}
        s = {}
        y = {}
        risk_cost = {}
        end_time = model.addVar("end_time", vtype="C", lb=0)

        for i in range(self.n_jobs):
            c[i] = model.addVar(f"c{i}", vtype="C", lb=0)
            s[i] = model.addVar(f"s{i}", vtype="C", lb=0)
            risk_cost[i] = model.addVar(f"risk_cost-{i}", vtype="C", lb=0)

        for i in range(self.n_jobs):
            for k in range(self.n_jobs):
                if i != k:
                    y[i, k] = model.addVar(f"y_{i}_{k}", vtype="B")

        # Stochastic job completion and start time constraints
        for i in range(self.n_jobs):
            model.addCons(c[i] >= s[i] + processing_times[i], f"time_{i}")

        for job_before, job_after in precedence_constraints:
            model.addCons(s[job_after] >= c[job_before], f"prec_{job_before}_to_{job_after}")

        for i in range(self.n_jobs):
            model.addCons(end_time >= c[i], f"makespan_{i}")

        for i in range(self.n_jobs):
            model.addCons(risk_cost[i] == tariff_impacts[i] * processing_times[i], f"risk_cost_{i}")

        M = 1e6
        for j in range(self.n_machines):
            jobs_on_machine = [i for i in range(self.n_jobs) if machine_assignment[i] == j]
            for i in range(len(jobs_on_machine)):
                for k in range(i + 1, len(jobs_on_machine)):
                    job_i = jobs_on_machine[i]
                    job_k = jobs_on_machine[k]
                    model.addCons(s[job_k] >= c[job_i] - M * (1 - y[job_i, job_k]), f"seq_{job_i}_{job_k}_machine{j}")
                    model.addCons(s[job_i] >= c[job_k] - M * y[job_i, job_k], f"seq_{job_k}_{job_i}_machine{j}")
                    model.addCons(y[job_i, job_k] + y[job_k, job_i] == 1, f"order_{job_i}_{job_k}")

        objective_expr = end_time + self.risk_weight * quicksum(risk_cost[i] for i in range(self.n_jobs))
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        return model.getStatus(), solve_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_jobs': 1500,
        'n_machines': 420,
        'n_groups': 180,
        'tariff_mean': 105.0,
        'tariff_std': 0.66,
        'risk_weight': 0.8,
    }

    job_scheduling = JobSchedulingRobust(parameters)
    instance = job_scheduling.generate_instance()
    solve_status, solve_time = job_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")