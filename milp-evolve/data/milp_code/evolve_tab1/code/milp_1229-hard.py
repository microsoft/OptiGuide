import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NeighborhoodMaintenanceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.num_teams > 0 and self.num_neighborhoods > 0
        assert self.min_maintenance_cost >= 0 and self.max_maintenance_cost >= self.min_maintenance_cost
        assert self.min_security_cost >= 0 and self.max_security_cost >= self.min_security_cost
        assert self.min_task_capacity > 0 and self.max_task_capacity >= self.min_task_capacity
        assert self.min_coverage_tasks >= 0 and self.max_coverage_tasks >= self.min_coverage_tasks
        
        maintenance_costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost + 1, self.num_teams)
        security_costs = np.random.randint(self.min_security_cost, self.max_security_cost + 1, (self.num_teams, self.num_neighborhoods))
        task_capacities = np.random.randint(self.min_task_capacity, self.max_task_capacity + 1, self.num_teams)
        coverage_tasks = np.random.randint(self.min_coverage_tasks, self.max_coverage_tasks + 1, self.num_neighborhoods)
        
        priority_neighborhoods = np.random.choice([0, 1], self.num_neighborhoods, p=[0.7, 0.3]) # 30% high-priority neighborhoods
        
        return {
            "maintenance_costs": maintenance_costs,
            "security_costs": security_costs,
            "task_capacities": task_capacities,
            "coverage_tasks": coverage_tasks,
            "priority_neighborhoods": priority_neighborhoods
        }

    def solve(self, instance):
        maintenance_costs = instance["maintenance_costs"]
        security_costs = instance["security_costs"]
        task_capacities = instance["task_capacities"]
        coverage_tasks = instance["coverage_tasks"]
        priority_neighborhoods = instance["priority_neighborhoods"]

        model = Model("NeighborhoodMaintenanceOptimization")
        num_teams = len(maintenance_costs)
        num_neighborhoods = len(coverage_tasks)
        
        team_vars = {t: model.addVar(vtype="B", name=f"Team_{t}") for t in range(num_teams)}
        neighborhood_vars = {(t, n): model.addVar(vtype="C", name=f"Neighborhood_{t}_{n}") for t in range(num_teams) for n in range(num_neighborhoods)}
        unmet_security_vars = {n: model.addVar(vtype="C", name=f"Unmet_Neighborhood_{n}") for n in range(num_neighborhoods)}

        model.setObjective(
            quicksum(maintenance_costs[t] * team_vars[t] for t in range(num_teams)) +
            quicksum(security_costs[t][n] * neighborhood_vars[t, n] for t in range(num_teams) for n in range(num_neighborhoods)) +
            quicksum(50000 * unmet_security_vars[n] for n in range(num_neighborhoods) if priority_neighborhoods[n] == 1),
            "minimize"
        )

        # Coverage tasks satisfaction
        for n in range(num_neighborhoods):
            model.addCons(quicksum(neighborhood_vars[t, n] for t in range(num_teams)) + unmet_security_vars[n] == coverage_tasks[n], f"CoverageRequirement_{n}")
        
        # Task capacity limits for each team
        for t in range(num_teams):
            model.addCons(quicksum(neighborhood_vars[t, n] for n in range(num_neighborhoods)) <= task_capacities[t] * team_vars[t], f"MaintenanceDone_{t}")

        # Neighborhood tasks only if team is active
        for t in range(num_teams):
            for n in range(num_neighborhoods):
                model.addCons(neighborhood_vars[t, n] <= coverage_tasks[n] * team_vars[t], f"ActiveTeamConstraint_{t}_{n}")

        # Priority neighborhood coverage
        for n in range(num_neighborhoods):
            if priority_neighborhoods[n] == 1:
                model.addCons(quicksum(neighborhood_vars[t, n] for t in range(num_teams)) + unmet_security_vars[n] >= coverage_tasks[n], f"SecurityGuarantee_{n}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == "__main__":
    seed = 42
    parameters = {
        'num_teams': 450,
        'num_neighborhoods': 100,
        'min_maintenance_cost': 3000,
        'max_maintenance_cost': 3000,
        'min_security_cost': 450,
        'max_security_cost': 500,
        'min_task_capacity': 2000,
        'max_task_capacity': 2000,
        'min_coverage_tasks': 750,
        'max_coverage_tasks': 800,
    }

    optimizer = NeighborhoodMaintenanceOptimization(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")