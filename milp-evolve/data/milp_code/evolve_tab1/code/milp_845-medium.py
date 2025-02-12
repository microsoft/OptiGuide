import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CloudTaskScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def task_processing_times(self):
        base_processing_time = 10.0  # base processing time in units
        return base_processing_time * np.random.rand(self.n_tasks, self.n_servers)

    def generate_instance(self):
        arrival_rates = self.randint(self.n_tasks, self.arrival_rate_interval)
        server_capacities = self.randint(self.n_servers, self.server_capacity_interval)
        activation_costs = self.randint(self.n_servers, self.activation_cost_interval)
        task_processing_times = self.task_processing_times()

        server_capacities = server_capacities * self.ratio * np.sum(arrival_rates) / np.sum(server_capacities)
        server_capacities = np.round(server_capacities)
        
        job_priorities = self.randint(self.n_tasks, (0, 2))  # Binary {0,1} priority

        res = {
            'arrival_rates': arrival_rates,
            'server_capacities': server_capacities,
            'activation_costs': activation_costs,
            'task_processing_times': task_processing_times,
            'job_priorities': job_priorities
        }

        return res

    def solve(self, instance):
        # Instance data
        arrival_rates = instance['arrival_rates']
        server_capacities = instance['server_capacities']
        activation_costs = instance['activation_costs']
        task_processing_times = instance['task_processing_times']
        job_priorities = instance['job_priorities']

        n_tasks = len(arrival_rates)
        n_servers = len(server_capacities)

        model = Model("CloudTaskScheduling")

        # Decision variables
        activate_server = {j: model.addVar(vtype="B", name=f"Activate_{j}") for j in range(n_servers)}
        assign_task = {(i, j): model.addVar(vtype="B", name=f"Assign_{i}_{j}") for i in range(n_tasks) for j in range(n_servers)}
        high_priority_activate = {j: model.addVar(vtype="B", name=f"HighPriorityActivate_{j}") for j in range(n_servers)}

        # Objective: Minimize the total cost including activation and penalty for wait time
        penalty_per_wait_time = 50
        high_priority_penalty = 100  # Extra penalty for high-priority tasks
        objective_expr = quicksum(activation_costs[j] * activate_server[j] for j in range(n_servers)) + penalty_per_wait_time * quicksum(task_processing_times[i, j] * assign_task[i, j] for i in range(n_tasks) for j in range(n_servers)) + quicksum(high_priority_penalty * high_priority_activate[j] for j in range(n_servers) if any(job_priorities[i] for i in range(n_tasks)))

        # Constraints: each task must be assigned to exactly one server
        for i in range(n_tasks):
            model.addCons(quicksum(assign_task[i, j] for j in range(n_servers)) == 1, f"Task_Assignment_{i}")
        
        # Constraints: server capacity limits must be respected
        for j in range(n_servers):
            model.addCons(quicksum(arrival_rates[i] * assign_task[i, j] for i in range(n_tasks)) <= server_capacities[j] * activate_server[j], f"Server_Capacity_{j}")

        # Constraint: Average wait time minimized (All tasks processed within permissible limits)
        for i in range(n_tasks):
            for j in range(n_servers):
                model.addCons(task_processing_times[i, j] * assign_task[i, j] <= activate_server[j] * 100, f"Service_Time_Limit_{i}_{j}")

        # New Constraints:
        for i in range(n_tasks):
            if job_priorities[i]:  # For high-priority tasks
                model.addCons(quicksum(assign_task[i, j] for j in range(n_servers) if high_priority_activate[j]) >= 1, f"HighPriorityCover_{i}")
                for j in range(n_servers):
                    model.addCons(high_priority_activate[j] <= activate_server[j], f"HighPriorityLink_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_tasks': 100,
        'n_servers': 50,
        'arrival_rate_interval': (2, 20),
        'server_capacity_interval': (450, 1350),
        'activation_cost_interval': (200, 400),
        'ratio': 45.0,
    }

    task_scheduling = CloudTaskScheduling(parameters, seed=seed)
    instance = task_scheduling.generate_instance()
    solve_status, solve_time = task_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")