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

        # Cost matrices
        team_costs = np.random.randint(10, 100, size=(n_teams, n_locations))
        activation_costs = np.random.randint(20, 200, size=n_locations)

        # Capacities and demands
        transportation_capacity = np.random.randint(50, 200, size=n_locations)
        team_demand = np.random.randint(5, 20, size=n_teams)

        res = {
            'n_teams': n_teams,
            'n_locations': n_locations,
            'team_costs': team_costs,
            'activation_costs': activation_costs,
            'transportation_capacity': transportation_capacity,
            'team_demand': team_demand
        }
        # New data based on convex hull formulation
        resource_limits = np.random.randint(200, 500, size=(n_teams, n_locations))
        res['resource_limits'] = resource_limits
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_teams = instance['n_teams']
        n_locations = instance['n_locations']
        team_costs = instance['team_costs']
        activation_costs = instance['activation_costs']
        transportation_capacity = instance['transportation_capacity']
        team_demand = instance['team_demand']
        resource_limits = instance['resource_limits']

        model = Model("SimplifiedTeamDeployment")

        # Variables
        x = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(n_locations)}

        # New resource allocation variables
        u = {}
        for i in range(n_teams):
            for j in range(n_locations):
                u[i, j] = model.addVar(vtype="I", lb=0, ub=resource_limits[i, j], name=f"u_{i}_{j}")

        # Objective function: Minimize total cost
        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(z[j] * activation_costs[j] for j in range(n_locations))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) == 1, name=f"team_assignment_{i}")

        # Capacities constraints and logical constraint for activation based on minimum assignments
        min_teams_per_location = 5  # Minimum number of teams to activate a location
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * team_demand[i] for i in range(n_teams)) <= transportation_capacity[j] * z[j],
                          name=f"transportation_capacity_{j}")

            # Logical condition constraint: Minimum teams assigned to activate location
            model.addCons(quicksum(x[i, j] for i in range(n_teams)) >= min_teams_per_location * z[j],
                          name=f"min_teams_to_activate_{j}")

        # Convex hull related constraints
        for i in range(n_teams):
            for j in range(n_locations):
                model.addCons(u[i, j] <= resource_limits[i, j] * x[i, j], name=f"resource_bound_{i}_{j}")

                model.addCons(u[i, j] >= 0, name=f"resource_min_{i}_{j}")

        # Ensure the sum of resources allocated does not exceed availability
        for j in range(n_locations):
            model.addCons(quicksum(u[i, j] for i in range(n_teams)) <= transportation_capacity[j] * z[j], name=f"resource_allocation_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 60,
        'max_teams': 672,
        'min_locations': 12,
        'max_locations': 900,
        'min_teams_per_location': 250,
    }
    deployment = SimplifiedTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")