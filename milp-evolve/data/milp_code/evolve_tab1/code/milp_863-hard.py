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
        
        # New data for resources and maintenance costs
        resource_limits = self.randint(self.n_servers, self.resource_limit_interval)
        maintenance_costs = self.randint(self.n_servers, self.maintenance_cost_interval)

        # New data for carbon emissions and renewable energy costs
        carbon_emissions = self.randint(self.n_servers, self.carbon_emission_interval)
        renewable_energy_costs = self.randint(self.n_servers, self.renewable_cost_interval)

        res = {
            'arrival_rates': arrival_rates,
            'server_capacities': server_capacities,
            'activation_costs': activation_costs,
            'task_processing_times': task_processing_times,
            'resource_limits': resource_limits,
            'maintenance_costs': maintenance_costs,
            'carbon_emissions': carbon_emissions,
            'renewable_energy_costs': renewable_energy_costs
        }
        
        return res

    def solve(self, instance):
        # Instance data
        arrival_rates = instance['arrival_rates']
        server_capacities = instance['server_capacities']
        activation_costs = instance['activation_costs']
        task_processing_times = instance['task_processing_times']
        resource_limits = instance['resource_limits']
        maintenance_costs = instance['maintenance_costs']
        carbon_emissions = instance['carbon_emissions']
        renewable_energy_costs = instance['renewable_energy_costs']

        n_tasks = len(arrival_rates)
        n_servers = len(server_capacities)

        model = Model("CloudTaskScheduling")

        # Decision variables
        activate_server = {j: model.addVar(vtype="B", name=f"Activate_{j}") for j in range(n_servers)}
        assign_task = {(i, j): model.addVar(vtype="B", name=f"Assign_{i}_{j}") for i in range(n_tasks) for j in range(n_servers)}
        resource_use = {j: model.addVar(vtype="C", name=f"ResourceUse_{j}") for j in range(n_servers)}
        
        # New decision variables
        renewable_energy_var = {j: model.addVar(vtype="B", name=f"Renewable_{j}") for j in range(n_servers)}
        inventory_level = {j: model.addVar(vtype="C", name=f"Inventory_{j}") for j in range(n_servers)}

        # Objective: Minimize the total cost including activation, wait time penalty, maintenance cost, renewable energy cost, and carbon emission cost
        penalty_per_wait_time = 50
        objective_expr = quicksum(activation_costs[j] * activate_server[j] for j in range(n_servers)) + \
                         penalty_per_wait_time * quicksum(task_processing_times[i, j] * assign_task[i, j] for i in range(n_tasks) for j in range(n_servers)) + \
                         quicksum(maintenance_costs[j] * resource_use[j] for j in range(n_servers)) + \
                         quicksum(renewable_energy_costs[j] * renewable_energy_var[j] for j in range(n_servers)) + \
                         quicksum(carbon_emissions[j] * activate_server[j] for j in range(n_servers))

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

        # New Constraints: Resource usage must be within limits
        for j in range(n_servers):
            model.addCons(resource_use[j] <= resource_limits[j], f"Resource_Use_Limit_{j}")
            
        # Renewable energy percentage constraint
        model.addCons(quicksum(renewable_energy_var[j] for j in range(n_servers)) >= self.renewable_percentage * n_servers, f"Renewable_Energy_Percentage")
        
        # Carbon emission constraint
        model.addCons(quicksum(carbon_emissions[j] * activate_server[j] for j in range(n_servers)) <= self.carbon_limit, f"Carbon_Limit")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_tasks': 200,
        'n_servers': 27,
        'arrival_rate_interval': (6, 60),
        'server_capacity_interval': (112, 337),
        'activation_cost_interval': (500, 1000),
        'ratio': 337.5,
        'resource_limit_interval': (10, 50),
        'maintenance_cost_interval': (555, 2250),
        'carbon_emission_interval': (10, 30),
        'renewable_cost_interval': (25, 50),
        'carbon_limit': 2000,
        'renewable_percentage': 0.73,
    }

    task_scheduling = CloudTaskScheduling(parameters, seed=seed)
    instance = task_scheduling.generate_instance()
    solve_status, solve_time = task_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")