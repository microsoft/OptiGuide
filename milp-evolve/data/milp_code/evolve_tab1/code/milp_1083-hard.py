import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class PowerEfficientTaskDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def get_instance(self):
        assert self.n_data_centers > 0 and self.n_tasks >= self.n_data_centers
        assert self.min_power_cost >= 0 and self.max_power_cost >= self.min_power_cost
        assert self.min_bandwidth_cost >= 0 and self.max_bandwidth_cost >= self.min_bandwidth_cost
        assert self.min_power_capacity > 0 and self.max_power_capacity >= self.min_power_capacity
        
        power_costs = np.random.randint(self.min_power_cost, self.max_power_cost + 1, self.n_data_centers)
        bandwidth_costs = np.random.randint(self.min_bandwidth_cost, self.max_bandwidth_cost + 1, (self.n_data_centers, self.n_tasks))
        power_capacities = np.random.randint(self.min_power_capacity, self.max_power_capacity + 1, self.n_data_centers)
        computational_rewards = np.random.uniform(10, 100, self.n_tasks)
        power_usage = np.random.uniform(0.5, 2.0, self.n_data_centers).tolist()
        
        # Normalize computational rewards to make the objective function simpler
        computational_rewards = computational_rewards / np.max(computational_rewards)
        
        return {
            "power_costs": power_costs,
            "bandwidth_costs": bandwidth_costs,
            "power_capacities": power_capacities,
            "computational_rewards": computational_rewards,
            "power_usage": power_usage
        }

    def solve(self, instance):
        power_costs = instance['power_costs']
        bandwidth_costs = instance['bandwidth_costs']
        power_capacities = instance['power_capacities']
        computational_rewards = instance['computational_rewards']
        power_usage = instance['power_usage']

        model = Model("PowerEfficientTaskDistribution")
        n_data_centers = len(power_costs)
        n_tasks = len(bandwidth_costs[0])
        
        data_center_vars = {dc: model.addVar(vtype="B", name=f"DataCenter_{dc}") for dc in range(n_data_centers)}
        task_vars = {(dc, t): model.addVar(vtype="B", name=f"DataCenter_{dc}_Task_{t}") for dc in range(n_data_centers) for t in range(n_tasks)}
        power_vars = {dc: model.addVar(vtype="C", name=f"Power_{dc}", lb=0) for dc in range(n_data_centers)}

        model.setObjective(
            quicksum(computational_rewards[t] * task_vars[dc, t] for dc in range(n_data_centers) for t in range(n_tasks)) -
            quicksum(power_costs[dc] * data_center_vars[dc] for dc in range(n_data_centers)) -
            quicksum(bandwidth_costs[dc][t] * task_vars[dc, t] for dc in range(n_data_centers) for t in range(n_tasks)) -
            quicksum(power_vars[dc] * power_usage[dc] for dc in range(n_data_centers)),
            "maximize"
        )

        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[dc, t] for dc in range(n_data_centers)) == 1, f"Task_{t}_Assignment")
        
        for dc in range(n_data_centers):
            for t in range(n_tasks):
                model.addCons(task_vars[dc, t] <= data_center_vars[dc], f"DataCenter_{dc}_Service_{t}")
        
        for dc in range(n_data_centers):
            model.addCons(quicksum(task_vars[dc, t] for t in range(n_tasks)) <= power_capacities[dc], f"DataCenter_{dc}_Capacity")

        for dc in range(n_data_centers):
            model.addCons(power_vars[dc] == quicksum(task_vars[dc, t] for t in range(n_tasks)) * power_usage[dc], f"PowerUsage_{dc}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_data_centers': 50,
        'n_tasks': 300,
        'min_power_cost': 525,
        'max_power_cost': 2500,
        'min_bandwidth_cost': 350,
        'max_bandwidth_cost': 1000,
        'min_power_capacity': 90,
        'max_power_capacity': 600,
    }
    
    task_optimizer = PowerEfficientTaskDistribution(parameters, seed=seed)
    instance = task_optimizer.get_instance()
    solve_status, solve_time, objective_value = task_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")