import random
import time
import numpy as np
from scipy.stats import poisson, binom
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
        team_costs = poisson.rvs(mu=50, size=(n_teams, n_locations))
        equipment_costs = binom.rvs(n=50, p=0.5, size=(n_equipment, n_locations))
        activation_costs = np.random.normal(loc=100, scale=20, size=n_locations)

        # Capacities and demands
        transportation_capacity = np.random.gamma(shape=2.0, scale=50.0, size=n_locations)
        communication_capacity = np.random.gamma(shape=2.0, scale=30.0, size=n_locations)
        team_demand = poisson.rvs(mu=10, size=n_teams)
        equipment_demand = poisson.rvs(mu=8, size=n_equipment)

        # Distances for nonlinear penalty function
        distances = np.random.randint(1, 100, size=n_locations)
        nonlinearity_factor = np.random.randint(1, 5, size=n_locations)

        # Traffic and Maintenance Data
        traffic_conditions = np.random.randint(0, 2, size=n_locations)   # 0 for normal, 1 for high traffic
        maintenance_intervals = np.random.randint(5, 15, size=n_teams)   # Maintenance interval in number of trips

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
            'distances': distances,
            'nonlinearity_factor': nonlinearity_factor,
            'traffic_conditions': traffic_conditions,
            'maintenance_intervals': maintenance_intervals
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
        distances = instance['distances']
        nonlinearity_factor = instance['nonlinearity_factor']
        traffic_conditions = instance['traffic_conditions']
        maintenance_intervals = instance['maintenance_intervals']

        model = Model("FieldTeamDeployment")

        # Variables
        x = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        y = {}
        for k in range(n_equipment):
            for j in range(n_locations):
                y[k, j] = model.addVar(vtype="B", name=f"y_{k}_{j}")

        z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(n_locations)}

        # Nonlinear penalty variables
        p = {j: model.addVar(vtype="C", name=f"p_{j}") for j in range(n_locations)}

        # Traffic penalty variables
        t_penalty = {j: model.addVar(vtype="C", name=f"t_penalty_{j}") for j in range(n_locations)}

        # Maintenance counter
        m_counter = {i: model.addVar(vtype="C", name=f"m_counter_{i}") for i in range(n_teams)}

        # Objective function: Minimize total cost
        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(y[k, j] * equipment_costs[k, j] for k in range(n_equipment) for j in range(n_locations)) + \
                     quicksum(z[j] * activation_costs[j] for j in range(n_locations)) + \
                     quicksum(p[j] for j in range(n_locations)) + \
                     quicksum(t_penalty[j] for j in range(n_locations))

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

            # Nonlinear penalty constraints
            model.addCons(p[j] >= distances[j] * z[j] ** nonlinearity_factor[j], name=f"nonlinear_penalty_{j}")

        for k in range(n_equipment):
            for j in range(n_locations):
                model.addCons(y[k, j] <= z[j], name=f"equipment_placement_{k}_{j}")

        # Traffic penalty constraints
        for j in range(n_locations):
            model.addCons(t_penalty[j] >= traffic_conditions[j] * z[j] * 100, name=f"traffic_penalty_{j}")

        # Maintenance constraints
        for i in range(n_teams):
            model.addCons(m_counter[i] == quicksum(x[i, j] for j in range(n_locations)), name=f"maintenance_counter_{i}")
            model.addCons(m_counter[i] <= maintenance_intervals[i], name=f"maintenance_interval_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 10,
        'max_teams': 100,
        'min_locations': 30,
        'max_locations': 160,
        'min_equipment': 6,
        'max_equipment': 350,
    }

    deployment = FieldTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")