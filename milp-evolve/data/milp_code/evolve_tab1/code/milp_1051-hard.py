import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class JobSchedulingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_machines > 0 and self.n_jobs > 0
        assert self.min_cost_machine >= 0 and self.max_cost_machine >= self.min_cost_machine
        assert self.min_cost_processing >= 0 and self.max_cost_processing >= self.min_cost_processing
        assert self.min_capacity_machine > 0 and self.max_capacity_machine >= self.min_capacity_machine
        assert self.min_processing_requirement >= 0 and self.max_processing_requirement >= self.min_processing_requirement

        fixed_costs = np.random.randint(self.min_cost_machine, self.max_cost_machine + 1, self.n_machines)
        processing_costs = np.random.randint(self.min_cost_processing, self.max_cost_processing + 1, (self.n_machines, self.n_jobs))
        capacities = np.random.randint(self.min_capacity_machine, self.max_capacity_machine + 1, self.n_machines)
        processing_requirements = np.random.randint(self.min_processing_requirement, self.max_processing_requirement + 1, self.n_jobs)
        penalty_costs = np.random.uniform(10, 50, self.n_jobs).tolist()

        return {
            "fixed_costs": fixed_costs,
            "processing_costs": processing_costs,
            "capacities": capacities,
            "processing_requirements": processing_requirements,
            "penalty_costs": penalty_costs,
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        processing_costs = instance['processing_costs']
        capacities = instance['capacities']
        processing_requirements = instance['processing_requirements']
        penalty_costs = instance['penalty_costs']

        model = Model("JobSchedulingOptimization")
        n_machines = len(fixed_costs)
        n_jobs = len(processing_costs[0])
        
        machine_vars = {m: model.addVar(vtype="B", name=f"Machine_{m}") for m in range(n_machines)}
        processing_vars = {(m, j): model.addVar(vtype="C", name=f"Processing_{m}_Job_{j}") for m in range(n_machines) for j in range(n_jobs)}
        unprocessed_job_vars = {j: model.addVar(vtype="C", name=f"Unprocessed_Job_{j}") for j in range(n_jobs)}

        # Objective function: Minimize total cost (fixed + processing + penalty for unprocessed time)
        model.setObjective(
            quicksum(fixed_costs[m] * machine_vars[m] for m in range(n_machines)) +
            quicksum(processing_costs[m][j] * processing_vars[m, j] for m in range(n_machines) for j in range(n_jobs)) +
            quicksum(penalty_costs[j] * unprocessed_job_vars[j] for j in range(n_jobs)),
            "minimize"
        )

        # Constraints
        # Processing satisfaction (total processing time and unprocessed time must cover total processing requirement)
        for j in range(n_jobs):
            model.addCons(quicksum(processing_vars[m, j] for m in range(n_machines)) + unprocessed_job_vars[j] == processing_requirements[j], f"Processing_Satisfaction_{j}")
        
        # Capacity limits for each machine
        for m in range(n_machines):
            model.addCons(quicksum(processing_vars[m, j] for j in range(n_jobs)) <= capacities[m] * machine_vars[m], f"Machine_Capacity_{m}")

        # Processing only if machine is operational
        for m in range(n_machines):
            for j in range(n_jobs):
                model.addCons(processing_vars[m, j] <= processing_requirements[j] * machine_vars[m], f"Operational_Constraint_{m}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_machines': 400,
        'n_jobs': 400,
        'min_cost_machine': 3000,
        'max_cost_machine': 5000,
        'min_cost_processing': 30,
        'max_cost_processing': 200,
        'min_capacity_machine': 1400,
        'max_capacity_machine': 2000,
        'min_processing_requirement': 50,
        'max_processing_requirement': 1600,
    }
    
    job_scheduler = JobSchedulingOptimization(parameters, seed=42)
    instance = job_scheduler.generate_instance()
    solve_status, solve_time, objective_value = job_scheduler.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")