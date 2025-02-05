import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedTeamDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_teams = random.randint(self.min_teams, self.max_teams)
        n_locations = random.randint(self.min_locations, self.max_locations)

        # Cost matrix
        team_costs = np.random.randint(10, 50, size=(n_teams, n_locations))  # simplified cost distribution

        # Capacities and demands
        transportation_capacity = np.random.randint(30, 150, size=n_locations)  # simplified capacity
        team_demand = np.random.randint(5, 15, size=n_teams)  # simplified team demand

        res = {
            'n_teams': n_teams,
            'n_locations': n_locations,
            'team_costs': team_costs,
            'transportation_capacity': transportation_capacity,
            'team_demand': team_demand
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_teams = instance['n_teams']
        n_locations = instance['n_locations']
        team_costs = instance['team_costs']
        transportation_capacity = instance['transportation_capacity']
        team_demand = instance['team_demand']

        model = Model("SimplifiedTeamDeployment")

        # Variables
        x = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Objective function: Minimize total cost
        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations))
        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) == 1, name=f"team_assignment_{i}")

        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * team_demand[i] for i in range(n_teams)) <= transportation_capacity[j],
                          name=f"transportation_capacity_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 300,
        'max_teams': 300,
        'min_locations': 7,
        'max_locations': 1000,
    }

    deployment = SimplifiedTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")