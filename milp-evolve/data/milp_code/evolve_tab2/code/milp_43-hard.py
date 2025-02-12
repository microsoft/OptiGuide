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
        processing_times = [random.randint(1, 50) for _ in range(self.n_jobs)] 
        precedence_constraints = self.create_random_precedence_constraints(self.n_jobs, self.n_groups)
        machine_assignment = [random.randint(0, self.n_machines - 1) for _ in range(self.n_jobs)]
        tariff_impacts = np.random.normal(self.tariff_mean, self.tariff_std, size=self.n_jobs)
        machine_capacity = np.random.randint(100, 200, size=self.n_machines)
        setup_times = np.random.randint(5, 15, size=(self.n_jobs, self.n_jobs))
        energy_consumption = np.random.uniform(1.0, 3.0, size=self.n_jobs)

        res = {
            'processing_times': processing_times,
            'precedence_constraints': precedence_constraints,
            'machine_assignment': machine_assignment,
            'tariff_impacts': tariff_impacts,
            'machine_capacity': machine_capacity,
            'setup_times': setup_times,
            'energy_consumption': energy_consumption
        }
        return res

    def solve(self, instance):
        processing_times = instance['processing_times']
        precedence_constraints = instance['precedence_constraints']
        machine_assignment = instance['machine_assignment']
        tariff_impacts = instance['tariff_impacts']
        machine_capacity = instance['machine_capacity']
        setup_times = instance['setup_times']
        energy_consumption = instance['energy_consumption']

        model = Model("AdvancedJobShopScheduling")
        
        c = {}  # Completion time variables for each job
        s = {}  # Start time variables for each job on its assigned machine
        y = {}  # Binary variables for sequencing jobs
        risk_cost = {}  # Additional cost variables due to trade tariffs
        energy_cost = {}  # Energy consumption cost variables
        aux_breakdown = {}  # Auxillary binary variables for machine breakdown management
        end_time = model.addVar("end_time", vtype="C", lb=0)

        for i in range(self.n_jobs):
            c[i] = model.addVar(f"c{i}", vtype="C", lb=0)
            s[i] = model.addVar(f"s{i}", vtype="C", lb=0)
            risk_cost[i] = model.addVar(f"risk_cost_{i}", vtype="C", lb=0)
            energy_cost[i] = model.addVar(f"energy_cost_{i}", vtype="C", lb=0)
            aux_breakdown[i] = model.addVar(f"aux_breakdown_{i}", vtype="B")

        for i in range(self.n_jobs):
            for k in range(self.n_jobs):
                if i != k:
                    y[i, k] = model.addVar(f"y_{i}_{k}", vtype="B")

        # Job completion and start time constraints
        for i in range(self.n_jobs):
            model.addCons(c[i] >= s[i] + processing_times[i], f"time_{i}")

        # Setup time constraints
        for i in range(self.n_jobs):
            for k in range(self.n_jobs):
                if i != k and machine_assignment[i] == machine_assignment[k]:
                    model.addCons(s[k] >= c[i] + setup_times[i][k] - 1e6 * (1 - y[i, k]), f"setup_{i}_{k}")

        # Precedence constraints
        for job_before, job_after in precedence_constraints:
            model.addCons(s[job_after] >= c[job_before], f"prec_{job_before}_to_{job_after}")

        # Makespan constraints
        for i in range(self.n_jobs):
            model.addCons(end_time >= c[i], f"makespan_{i}")

        # Risk cost and energy consumption constraints
        for i in range(self.n_jobs):
            model.addCons(risk_cost[i] == tariff_impacts[i] * processing_times[i], f"risk_cost_{i}")
            model.addCons(energy_cost[i] == energy_consumption[i] * processing_times[i], f"energy_cost_{i}")

        # Sequencing constraints on the same machine
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

        # Machine breakdown auxiliary constraint
        for j in range(self.n_machines):
            model.addCons(
                quicksum(processing_times[i] * (1 - aux_breakdown[i]) for i in range(self.n_jobs) if machine_assignment[i] == j) 
                <= machine_capacity[j], f"Machine_Capacity_{j}"
            )

        # New objective includes minimizing energy cost and setup time penalties
        setup_penalty = 30
        breakdown_penalty = 100
        objective_expr = end_time + self.risk_weight * quicksum(risk_cost[i] for i in range(self.n_jobs)) + \
                         self.energy_weight * quicksum(energy_cost[i] for i in range(self.n_jobs)) + \
                         setup_penalty * quicksum(setup_times[i][k] * y[i, k] for i in range(self.n_jobs) for k in range(self.n_jobs) if i != k) + \
                         breakdown_penalty * quicksum(aux_breakdown[i] for i in range(self.n_jobs))
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_jobs': 750,
        'n_machines': 840,
        'n_groups': 175,
        'tariff_mean': 189.0,
        'tariff_std': 0.73,
        'risk_weight': 0.73,
        'energy_weight': 0.31,
    }
    job_scheduling = JobScheduling(parameters)
    instance = job_scheduling.generate_instance()
    solve_status, solve_time = job_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")