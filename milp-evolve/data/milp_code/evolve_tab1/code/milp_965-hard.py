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
        
        # Data for resources and maintenance costs
        resource_limits = self.randint(self.n_servers, self.resource_limit_interval)
        maintenance_costs = self.randint(self.n_servers, self.maintenance_cost_interval)

        energy_limits = self.randint(self.n_servers, self.energy_limit_interval)
        carbon_limits = self.randint(self.n_servers, self.carbon_limit_interval)
        renewable_energy_costs = self.randint(self.n_servers, self.renewable_energy_cost_interval)

        res = {
            'arrival_rates': arrival_rates,
            'server_capacities': server_capacities,
            'activation_costs': activation_costs,
            'task_processing_times': task_processing_times,
            'resource_limits': resource_limits,
            'maintenance_costs': maintenance_costs,
            'energy_limits': energy_limits,
            'carbon_limits': carbon_limits,
            'renewable_energy_costs': renewable_energy_costs
        }
        
        # Generate task groups
        n_groups = min(self.n_tasks, self.n_servers * 2)
        task_groups = {g: np.random.choice(self.n_tasks, size=random.randint(5, 15), replace=False) for g in range(n_groups)}
        res['task_groups'] = task_groups

        return res

    def solve(self, instance):
        # Instance data
        arrival_rates = instance['arrival_rates']
        server_capacities = instance['server_capacities']
        activation_costs = instance['activation_costs']
        task_processing_times = instance['task_processing_times']
        resource_limits = instance['resource_limits']
        maintenance_costs = instance['maintenance_costs']
        energy_limits = instance['energy_limits']
        carbon_limits = instance['carbon_limits']
        renewable_energy_costs = instance['renewable_energy_costs']
        task_groups = instance['task_groups']

        n_tasks = len(arrival_rates)
        n_servers = len(server_capacities)
        n_groups = len(task_groups)

        model = Model("CloudTaskScheduling")

        # Decision variables
        activate_server = {j: model.addVar(vtype="B", name=f"Activate_{j}") for j in range(n_servers)}
        assign_task = {(i, j): model.addVar(vtype="B", name=f"Assign_{i}_{j}") for i in range(n_tasks) for j in range(n_servers)}
        resource_use = {j: model.addVar(vtype="C", name=f"ResourceUse_{j}") for j in range(n_servers)}
        energy_use = {j: model.addVar(vtype="C", name=f"EnergyUse_{j}") for j in range(n_servers)}
        renewable_energy_use = {j: model.addVar(vtype="B", name=f"RenewableEnergyUse_{j}") for j in range(n_servers)}
        carbon_emissions = {j: model.addVar(vtype="C", name=f"CarbonEmissions_{j}") for j in range(n_servers)}
        task_group = {g: model.addVar(vtype="B", name=f"TaskGroup_{g}") for g in range(n_groups)}

        # Objective: Minimize the total cost including activation, wait time penalty, maintenance cost, and energy cost
        penalty_per_wait_time = 50
        energy_cost = 200
        carbon_penalty = 100

        objective_expr = quicksum(activation_costs[j] * activate_server[j] for j in range(n_servers)) + \
                         penalty_per_wait_time * quicksum(task_processing_times[i, j] * assign_task[i, j] for i in range(n_tasks) for j in range(n_servers)) + \
                         quicksum(maintenance_costs[j] * resource_use[j] for j in range(n_servers)) + \
                         quicksum(energy_cost * energy_use[j] for j in range(n_servers)) + \
                         quicksum(carbon_penalty * carbon_emissions[j] for j in range(n_servers))

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

        # Constraints: Resource usage must be within limits
        for j in range(n_servers):
            model.addCons(resource_use[j] <= resource_limits[j], f"Resource_Use_Limit_{j}")

        # Constraints: Energy usage limits
        for j in range(n_servers):
            model.addCons(energy_use[j] <= energy_limits[j], f"Energy_Use_Limit_{j}")

        # Constraints: Renewable energy usage
        for j in range(n_servers):
            model.addCons(energy_use[j] * renewable_energy_use[j] >= self.renewable_threshold, f"Renewable_Energy_Use_{j}")

        # Constraints: Carbon emissions must be below limits
        for j in range(n_servers):
            model.addCons(carbon_emissions[j] <= carbon_limits[j], f"Carbon_Limit_{j}")

        # New constraints: Task groups must respect server capacity, energy use, and carbon emissions
        for g, tasks in task_groups.items():
            model.addCons(quicksum(assign_task[i, j] for i in tasks for j in range(n_servers)) == task_group[g] * len(tasks), f"Task_Group_Assignment_{g}")
            model.addCons(quicksum(arrival_rates[i] * assign_task[i, j] for i in tasks for j in range(n_servers)) <= quicksum(server_capacities[j] for j in range(n_servers)) * task_group[g], f"Task_Group_Capacity_{g}")
            model.addCons(quicksum(energy_use[j] for j in range(n_servers)) <= quicksum(energy_limits[j] for j in range(n_servers)) * task_group[g], f"Task_Group_Energy_{g}")
            model.addCons(quicksum(carbon_emissions[j] for j in range(n_servers)) <= quicksum(carbon_limits[j] for j in range(n_servers)) * task_group[g], f"Task_Group_Carbon_{g}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_tasks': 150,
        'n_servers': 60,
        'arrival_rate_interval': (84, 945),
        'server_capacity_interval': (336, 1010),
        'activation_cost_interval': (1000, 2000),
        'ratio': 506.25,
        'resource_limit_interval': (450, 2250),
        'maintenance_cost_interval': (416, 1687),
        'energy_limit_interval': (200, 800),
        'carbon_limit_interval': (125, 500),
        'renewable_energy_cost_interval': (900, 2700),
        'renewable_threshold': 0.24,
    }

    task_scheduling = CloudTaskScheduling(parameters, seed=seed)
    instance = task_scheduling.generate_instance()
    solve_status, solve_time = task_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")