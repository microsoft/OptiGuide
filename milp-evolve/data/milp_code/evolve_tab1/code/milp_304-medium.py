import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum


class FieldTeamDeployment:
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
        n_equipment = random.randint(self.min_equipment, self.max_equipment)

        # Cost matrices
        team_costs = np.random.randint(10, 100, size=(n_teams, n_locations))
        equipment_costs = np.random.randint(5, 50, size=(n_equipment, n_locations))
        activation_costs = np.random.randint(20, 200, size=n_locations)

        # Capacities and demands
        transportation_capacity = np.random.randint(50, 200, size=n_locations)
        communication_capacity = np.random.randint(30, 150, size=n_locations)
        team_demand = np.random.randint(5, 20, size=n_teams)
        equipment_demand = np.random.randint(3, 15, size=n_equipment)

        # Additional capacities
        secondary_transportation_capacity = np.random.randint(30, 150, size=n_locations)
        secondary_communication_capacity = np.random.randint(20, 100, size=n_locations)

        res = {
            'n_teams': n_teams,
            'n_locations': n_locations,
            'n_equipment': n_equipment,
            'team_costs': team_costs,
            'equipment_costs': equipment_costs,
            'activation_costs': activation_costs,
            'transportation_capacity': transportation_capacity,
            'communication_capacity': communication_capacity,
            'team_demand': team_demand,
            'equipment_demand': equipment_demand,
            'secondary_transportation_capacity': secondary_transportation_capacity,
            'secondary_communication_capacity': secondary_communication_capacity
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_teams = instance['n_teams']
        n_locations = instance['n_locations']
        n_equipment = instance['n_equipment']
        team_costs = instance['team_costs']
        equipment_costs = instance['equipment_costs']
        activation_costs = instance['activation_costs']
        transportation_capacity = instance['transportation_capacity']
        communication_capacity = instance['communication_capacity']
        team_demand = instance['team_demand']
        equipment_demand = instance['equipment_demand']
        secondary_transportation_capacity = instance['secondary_transportation_capacity']
        secondary_communication_capacity = instance['secondary_communication_capacity']

        model = Model("FieldTeamDeployment")

        # Variables
        x = {}
        w = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")
                w[i, j] = model.addVar(vtype="B", name=f"w_{i}_{j}")

        y = {}
        v = {}
        for k in range(n_equipment):
            for j in range(n_locations):
                y[k, j] = model.addVar(vtype="B", name=f"y_{k}_{j}")
                v[k, j] = model.addVar(vtype="B", name=f"v_{k}_{j}")

        z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(n_locations)}

        # Objective function: Minimize total cost
        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(y[k, j] * equipment_costs[k, j] for k in range(n_equipment) for j in range(n_locations)) + \
                     quicksum(z[j] * activation_costs[j] for j in range(n_locations)) + \
                     quicksum(w[i, j] * team_costs[i, j] * 0.1 for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(v[k, j] * equipment_costs[k, j] * 0.1 for k in range(n_equipment) for j in range(n_locations))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) == 1, name=f"team_assignment_{i}")

        for k in range(n_equipment):
            model.addCons(quicksum(y[k, j] for j in range(n_locations)) == 1, name=f"equipment_assignment_{k}")

        # Capacities constraints
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * team_demand[i] for i in range(n_teams)) <= transportation_capacity[j] * z[j],
                          name=f"transportation_capacity_{j}")

            model.addCons(quicksum(y[k, j] * equipment_demand[k] for k in range(n_equipment)) <= communication_capacity[j] * z[j],
                          name=f"communication_capacity_{j}")

            # Secondary capacities constraints
            model.addCons(quicksum(w[i, j] * team_demand[i] for i in range(n_teams)) <= secondary_transportation_capacity[j] * z[j],
                          name=f"secondary_transportation_capacity_{j}")

            model.addCons(quicksum(v[k, j] * equipment_demand[k] for k in range(n_equipment)) <= secondary_communication_capacity[j] * z[j],
                          name=f"secondary_communication_capacity_{j}")

        for k in range(n_equipment):
            for j in range(n_locations):
                model.addCons(y[k, j] <= z[j], name=f"equipment_placement_{k}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 0,
        'max_teams': 180,
        'min_locations': 1,
        'max_locations': 450,
        'min_equipment': 6,
        'max_equipment': 2250,
        'penalty_coefficient': 0.17,
    }

    deployment = FieldTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")