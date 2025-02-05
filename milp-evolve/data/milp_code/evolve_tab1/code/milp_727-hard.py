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
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_jobs > 0 and self.n_machines > 0
        assert self.min_processing_time >= 0 and self.max_processing_time >= self.min_processing_time

        processing_times = np.random.randint(self.min_processing_time, self.max_processing_time + 1, self.n_jobs)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_machines)
        
        return {
            "processing_times": processing_times,
            "capacities": capacities
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        processing_times = instance['processing_times']
        capacities = instance['capacities']
        
        model = Model("JobScheduling")
        n_jobs = len(processing_times)
        n_machines = len(capacities)
        
        # Decision variables
        job_vars = {(j, m): model.addVar(vtype="B", name=f"Job_{j}_Machine_{m}") for j in range(n_jobs) for m in range(n_machines)}
        
        # Completion time variable
        makespan = model.addVar(vtype="C", name="Makespan")
        
        # Objective: minimize the makespan
        model.setObjective(makespan, "minimize")
        
        # Constraints: Each job is assigned to exactly one machine
        for j in range(n_jobs):
            model.addCons(quicksum(job_vars[j, m] for m in range(n_machines)) == 1, f"Job_{j}_Assignment")

        # Constraints: Machine capacity should not exceed its limit
        for m in range(n_machines):
            model.addCons(quicksum(processing_times[j] * job_vars[j, m] for j in range(n_jobs)) <= capacities[m], f"Machine_{m}_Capacity")

        # Constraints: Makespan should be greater than or equal to the processing time of jobs assigned to any machine
        for m in range(n_machines):
            model.addCons(quicksum(processing_times[j] * job_vars[j, m] for j in range(n_jobs)) <= makespan, f"Machine_{m}_Makespan")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_jobs': 50,
        'n_machines': 90,
        'min_processing_time': 3,
        'max_processing_time': 600,
        'min_capacity': 2500,
        'max_capacity': 3000,
    }

    scheduler = JobScheduling(parameters, seed=42)
    instance = scheduler.generate_instance()
    solve_status, solve_time = scheduler.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")