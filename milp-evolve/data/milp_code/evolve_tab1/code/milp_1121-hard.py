import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DataCenterTaskAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_servers > 0 and self.n_tasks > 0
        assert self.min_cost_server >= 0 and self.max_cost_server >= self.min_cost_server
        assert self.min_cost_allocation >= 0 and self.max_cost_allocation >= self.min_cost_allocation
        assert self.min_capacity_server > 0 and self.max_capacity_server >= self.min_capacity_server
        assert self.min_task_demand >= 0 and self.max_task_demand >= self.min_task_demand

        server_usage_costs = np.random.randint(self.min_cost_server, self.max_cost_server + 1, self.n_servers)
        allocation_costs = np.random.randint(self.min_cost_allocation, self.max_cost_allocation + 1, (self.n_servers, self.n_tasks))
        capacities = np.random.randint(self.min_capacity_server, self.max_capacity_server + 1, self.n_servers)
        task_demands = np.random.randint(self.min_task_demand, self.max_task_demand + 1, self.n_tasks)
        no_allocation_penalties = np.random.uniform(100, 300, self.n_tasks).tolist()
        
        subsets = [random.sample(range(self.n_tasks), int(0.2 * self.n_tasks)) for _ in range(5)]
        min_coverage = np.random.randint(1, 5, 5)

        return {
            "server_usage_costs": server_usage_costs,
            "allocation_costs": allocation_costs,
            "capacities": capacities,
            "task_demands": task_demands,
            "no_allocation_penalties": no_allocation_penalties,
            "subsets": subsets,
            "min_coverage": min_coverage
        }

    def solve(self, instance):
        server_usage_costs = instance['server_usage_costs']
        allocation_costs = instance['allocation_costs']
        capacities = instance['capacities']
        task_demands = instance['task_demands']
        no_allocation_penalties = instance['no_allocation_penalties']
        subsets = instance['subsets']
        min_coverage = instance['min_coverage']

        model = Model("DataCenterTaskAllocation")
        n_servers = len(server_usage_costs)
        n_tasks = len(task_demands)
        
        server_vars = {s: model.addVar(vtype="B", name=f"Server_{s}") for s in range(n_servers)}
        task_allocation_vars = {(s, t): model.addVar(vtype="C", name=f"Server_{s}_Task_{t}") for s in range(n_servers) for t in range(n_tasks)}
        unmet_task_vars = {t: model.addVar(vtype="C", name=f"Unmet_Task_{t}") for t in range(n_tasks)}

        model.setObjective(
            quicksum(server_usage_costs[s] * server_vars[s] for s in range(n_servers)) +
            quicksum(allocation_costs[s][t] * task_allocation_vars[s, t] for s in range(n_servers) for t in range(n_tasks)) +
            quicksum(no_allocation_penalties[t] * unmet_task_vars[t] for t in range(n_tasks)),
            "minimize"
        )

        # Constraints
        # Task demand satisfaction (total allocations and unmet tasks must cover total demand)
        for t in range(n_tasks):
            model.addCons(quicksum(task_allocation_vars[s, t] for s in range(n_servers)) + unmet_task_vars[t] == task_demands[t], f"Task_Demand_Satisfaction_{t}")
        
        # Clique inequalities for server capacities
        for s in range(n_servers):
            for subset in subsets:
                model.addCons(quicksum(task_allocation_vars[s, t] for t in subset) <= capacities[s] * server_vars[s], f"Clique_Inequality_{s}_{tuple(subset)}")

        # Task allocation only if server is operational
        for s in range(n_servers):
            for t in range(n_tasks):
                model.addCons(task_allocation_vars[s, t] <= task_demands[t] * server_vars[s], f"Operational_Constraint_{s}_{t}")

        # Set covering constraints: ensure minimum number of critical tasks allocated by some server
        for i, subset in enumerate(subsets):
            model.addCons(quicksum(task_allocation_vars[s, t] for s in range(n_servers) for t in subset) >= min_coverage[i], f"Set_Covering_Constraint_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_servers': 900,
        'n_tasks': 60,
        'min_cost_server': 5000,
        'max_cost_server': 10000,
        'min_cost_allocation': 450,
        'max_cost_allocation': 600,
        'min_capacity_server': 1500,
        'max_capacity_server': 1800,
        'min_task_demand': 100,
        'max_task_demand': 1000,
    }

    optimizer = DataCenterTaskAllocation(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")