import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum


class MobileMedicalUnitsAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        n_units = random.randint(self.min_units, self.max_units)
        n_communities = random.randint(self.min_communities, self.max_communities)
        n_seasons = random.randint(self.min_seasons, self.max_seasons)

        # Cost matrices
        travel_costs = np.random.randint(50, 500, size=(n_units, n_communities))
        operation_costs = np.random.randint(200, 2000, size=n_units)
        seasonal_costs = np.random.randint(100, 1000, size=(n_units, n_seasons))

        # Capacities and demands
        unit_capacity = np.random.randint(100, 400, size=n_units)
        community_demand = np.random.randint(50, 300, size=(n_communities, n_seasons))

        res = {
            'n_units': n_units,
            'n_communities': n_communities,
            'n_seasons': n_seasons,
            'travel_costs': travel_costs,
            'operation_costs': operation_costs,
            'seasonal_costs': seasonal_costs,
            'unit_capacity': unit_capacity,
            'community_demand': community_demand
        }

        ### New instance data code ends here
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_units = instance['n_units']
        n_communities = instance['n_communities']
        n_seasons = instance['n_seasons']
        travel_costs = instance['travel_costs']
        operation_costs = instance['operation_costs']
        seasonal_costs = instance['seasonal_costs']
        unit_capacity = instance['unit_capacity']
        community_demand = instance['community_demand']

        model = Model("MobileMedicalUnitsAllocation")

        # Variables
        u_assign = {}
        for i in range(n_units):
            for j in range(n_communities):
                u_assign[i, j] = model.addVar(vtype="B", name=f"u_assign_{i}_{j}")

        u_active = {i: model.addVar(vtype="B", name=f"u_active_{i}") for i in range(n_units)}
        s_activate = {}
        for i in range(n_units):
            for s in range(n_seasons):
                s_activate[i, s] = model.addVar(vtype="B", name=f"s_activate_{i}_{s}")

        # Objective function: Minimize total cost
        total_cost = quicksum(u_assign[i, j] * travel_costs[i, j] for i in range(n_units) for j in range(n_communities)) + \
                     quicksum(u_active[i] * operation_costs[i] for i in range(n_units)) + \
                     quicksum(s_activate[i, s] * seasonal_costs[i, s] for i in range(n_units) for s in range(n_seasons))
        model.setObjective(total_cost, "minimize")

        # Constraints
        for j in range(n_communities):
            for s in range(n_seasons):
                model.addCons(quicksum(u_assign[i, j] * unit_capacity[i] * s_activate[i, s] for i in range(n_units)) >= community_demand[j, s], 
                              name=f"community_demand_{j}_{s}")

        for i in range(n_units):
            for s in range(n_seasons):
                model.addCons(s_activate[i, s] <= u_active[i], name=f"seasonal_operation_{i}_{s}")

            model.addCons(quicksum(u_assign[i, j] for j in range(n_communities)) >= u_active[i], name=f"unit_active_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_units': 0,
        'max_units': 25,
        'min_communities': 1,
        'max_communities': 300,
        'min_seasons': 12,
        'max_seasons': 12,
    }
    ### New parameter code ends here

    allocation = MobileMedicalUnitsAllocation(parameters, seed=seed)
    instance = allocation.get_instance()
    solve_status, solve_time = allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")