import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ComplexTeamDeployment:
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

        complex_team = np.random.poisson(lam=5, size=n_teams)
        
        # Introducing new data elements from AdvancedCombinatorialAuctionWithFLPMutualExclusivity
        n_facilities = n_locations
        env_impact_cost = np.random.uniform(1, 10, size=n_facilities).tolist()
        mutual_exclusivity_pairs = [(random.randint(0, n_facilities - 1), random.randint(0, n_facilities - 1)) for _ in range(self.n_exclusive_pairs)]
        
        res = {
            'n_teams': n_teams,
            'n_locations': n_locations,
            'team_costs': team_costs,
            'activation_costs': activation_costs,
            'transportation_capacity': transportation_capacity,
            'team_demand': team_demand,
            'complex_team': complex_team,
            'env_impact_cost': env_impact_cost,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_teams = instance['n_teams']
        n_locations = instance['n_locations']
        team_costs = instance['team_costs']
        activation_costs = instance['activation_costs']
        transportation_capacity = instance['transportation_capacity']
        team_demand = instance['team_demand']
        complex_team = instance['complex_team']
        env_impact_cost = instance['env_impact_cost']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']

        model = Model("ComplexTeamDeployment")

        # Variables
        x = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(n_locations)}
        env_impact_vars = {j: model.addVar(vtype="C", name=f"env_impact_{j}", lb=0) for j in range(n_locations)}

        facility_workload = {j: model.addVar(vtype="I", name=f"workload_{j}", lb=0) for j in range(n_locations)}

        # Objective function: Minimize total cost involving environmental costs
        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(z[j] * activation_costs[j] for j in range(n_locations)) + \
                     quicksum(env_impact_cost[j] * env_impact_vars[j] for j in range(n_locations))

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

            # Facility workload constraints
            model.addCons(facility_workload[j] == quicksum(x[i, j] * complex_team[i] for i in range(n_teams)), f"Workload_{j}")

            # Environmental impact constraints
            model.addCons(env_impact_vars[j] == quicksum(x[i, j] for i in range(n_teams)), f"EnvImpact_{j}")

        # Mutual exclusivity constraints
        for fac1, fac2 in mutual_exclusivity_pairs:
            model.addCons(z[fac1] + z[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 45,
        'max_teams': 336,
        'min_locations': 120,
        'max_locations': 675,
        'n_exclusive_pairs': 37,
    }

    deployment = ComplexTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")