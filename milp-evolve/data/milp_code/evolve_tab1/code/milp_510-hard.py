import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class TeamDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        n_teams = random.randint(self.min_teams, self.max_teams)
        n_locations = random.randint(self.min_locations, self.max_locations)

        team_costs = np.random.gamma(20, 2, size=(n_teams, n_locations))
        activation_costs = np.random.normal(100, 30, size=n_locations)
        transportation_capacity = np.random.randint(50, 200, size=n_locations)
        team_demand = np.random.randint(10, 25, size=n_teams)
        team_preferences = np.random.uniform(0, 1, size=(n_teams, n_locations))
        team_skills = np.random.randint(1, 10, size=n_teams)
        location_skill_requirements = np.random.randint(5, 15, size=n_locations)
        
        res = {
            'n_teams': n_teams,
            'n_locations': n_locations,
            'team_costs': team_costs,
            'activation_costs': activation_costs,
            'transportation_capacity': transportation_capacity,
            'team_demand': team_demand,
            'team_preferences': team_preferences,
            'team_skills': team_skills,
            'location_skill_requirements': location_skill_requirements
        }
        return res

    def solve(self, instance):
        n_teams = instance['n_teams']
        n_locations = instance['n_locations']
        team_costs = instance['team_costs']
        activation_costs = instance['activation_costs']
        transportation_capacity = instance['transportation_capacity']
        team_demand = instance['team_demand']
        team_preferences = instance['team_preferences']
        team_skills = instance['team_skills']
        location_skill_requirements = instance['location_skill_requirements']

        model = Model("TeamDeployment")

        x = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(n_locations)}

        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(z[j] * activation_costs[j] for j in range(n_locations))
        total_preference = quicksum(x[i, j] * team_preferences[i, j] for i in range(n_teams) for j in range(n_locations))

        model.setObjective(total_cost - total_preference, "minimize")

        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) <= 3, name=f"team_max_assignments_{i}")
        
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * team_demand[i] for i in range(n_teams)) <= transportation_capacity[j] * z[j],
                          name=f"transportation_capacity_{j}")
            model.addCons(quicksum(x[i, j] * team_skills[i] for i in range(n_teams)) >= location_skill_requirements[j] * z[j],
                          name=f"skill_requirement_{j}")
        
        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) >= 1, name=f"team_min_assignment_{i}")

        model.addCons(quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                      quicksum(z[j] * activation_costs[j] for j in range(n_locations)) <= self.budget, "BudgetConstraint")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 6,
        'max_teams': 330,
        'min_locations': 35,
        'max_locations': 1875,
        'max_assignments_per_team': 9,
        'budget': 100000,
    }

    deployment = TeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")