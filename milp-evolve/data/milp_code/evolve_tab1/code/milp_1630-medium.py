import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ProductionSchedulingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_machines > 0 and self.n_jobs > 0
        assert self.min_cost >= 0 and self.max_cost > self.min_cost
        assert self.min_time > 0 and self.max_time >= self.min_time
        assert self.min_raw_material_cost >= 0 and self.max_raw_material_cost >= self.min_raw_material_cost

        machine_costs = np.random.randint(self.min_cost, self.max_cost + 1, self.n_machines)
        job_times = np.random.randint(self.min_time, self.max_time + 1, (self.n_jobs, self.n_machines))
        raw_material_costs = np.random.uniform(self.min_raw_material_cost, self.max_raw_material_cost, self.n_jobs)
        job_disruption_risks = np.random.uniform(self.min_disruption_risk, self.max_disruption_risk, self.n_jobs)
        
        return {
            "machine_costs": machine_costs,
            "job_times": job_times,
            "raw_material_costs": raw_material_costs,
            "job_disruption_risks": job_disruption_risks,
        }

    def solve(self, instance):
        machine_costs = instance['machine_costs']
        job_times = instance['job_times']
        raw_material_costs = instance['raw_material_costs']
        job_disruption_risks = instance['job_disruption_risks']

        model = Model("ProductionSchedulingOptimization")

        n_machines = len(machine_costs)
        n_jobs = len(job_times)

        # Decision variables
        job_vars = {(j, m): model.addVar(vtype="B", name=f"Job_{j}_{m}") for j in range(n_jobs) for m in range(n_machines)}
        disruption_penalty = {j: model.addVar(vtype="C", name=f"DisruptionPenalty_{j}") for j in range(n_jobs)}

        # Objective: minimize the total cost including machine costs, raw material costs, and disruption penalties.
        model.setObjective(
            quicksum(machine_costs[m] * job_vars[j, m] for j in range(n_jobs) for m in range(n_machines)) +
            quicksum(raw_material_costs[j] for j in range(n_jobs)) +
            quicksum(disruption_penalty[j] * job_disruption_risks[j] for j in range(n_jobs)),
            "minimize"
        )
        
        # Constraints: Each job must be assigned to exactly one machine
        for j in range(n_jobs):
            model.addCons(
                quicksum(job_vars[j, m] for m in range(n_machines)) == 1,
                f"JobAssignment_{j}"
            )
        
        # Constraints: Machine capacity constraints
        for m in range(n_machines):
            model.addCons(
                quicksum(job_times[j, m] * job_vars[j, m] for j in range(n_jobs)) <= self.machine_capacity,
                f"MachineCapacity_{m}"
            )
        
        # Constraints: Disruption risk penalty constraints
        for j in range(n_jobs):
            model.addCons(
                disruption_penalty[j] >= quicksum(job_times[j, m] * job_vars[j, m] for m in range(n_machines)) * self.disruption_penalty_rate,
                f"DisruptionPenaltyConstraint_{j}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_machines': 30,
        'n_jobs': 200,
        'min_cost': 200,
        'max_cost': 3000,
        'min_time': 15,
        'max_time': 180,
        'min_raw_material_cost': 50.0,
        'max_raw_material_cost': 800.0,
        'min_disruption_risk': 0.1,
        'max_disruption_risk': 0.17,
        'machine_capacity': 200,
        'disruption_penalty_rate': 52.5,
    }

    optimizer = ProductionSchedulingOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")