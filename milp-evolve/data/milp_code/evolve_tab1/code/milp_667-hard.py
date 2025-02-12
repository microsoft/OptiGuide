import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class OptimizedHubPlacement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        # Randomly generate fixed opening costs for hubs
        opening_costs = np.random.randint(self.max_open_cost, size=self.n_hubs) + 1

        # Generate communication costs between each user and hub
        comm_costs = np.random.randint(self.max_comm_cost, size=(self.n_users, self.n_hubs)) + 1

        res = {
            'opening_costs': opening_costs,
            'comm_costs': comm_costs,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        opening_costs = instance['opening_costs']
        comm_costs = instance['comm_costs']

        model = Model("OptimizedHubPlacement")
        hub_open = {}
        hub_assign = {}

        # Create hub open variables and set fixed opening costs
        for h in range(self.n_hubs):
            hub_open[h] = model.addVar(vtype="B", name=f"HubOpen_{h}", obj=opening_costs[h])

        # Create hub assignment variables and set communication costs
        for u in range(self.n_users):
            for h in range(self.n_hubs):
                hub_assign[u, h] = model.addVar(vtype="B", name=f"HubAssign_{u}_{h}", obj=comm_costs[u, h])

        # Constraints: Ensure each user is assigned to exactly one hub
        for u in range(self.n_users):
            model.addCons(quicksum(hub_assign[u, h] for h in range(self.n_hubs)) == 1, f"UserAssign_{u}")

        # Constraints: Ensure a user can only be assigned to an open hub
        for u in range(self.n_users):
            for h in range(self.n_hubs):
                model.addCons(hub_assign[u, h] <= hub_open[h], f"UserToOpenHub_{u}_{h}")

        # Constraint: Limit the number of open hubs
        model.addCons(quicksum(hub_open[h] for h in range(self.n_hubs)) <= self.max_hubs, "MaxOpenHubs")

        # Set objective: Minimize total cost (fixed + communication costs)
        total_cost_expr = (quicksum(opening_costs[h] * hub_open[h] for h in range(self.n_hubs)) +
                           quicksum(comm_costs[u, h] * hub_assign[u, h] for u in range(self.n_users) for h in range(self.n_hubs)))

        model.setObjective(total_cost_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_users': 150,
        'n_hubs': 37,
        'max_open_cost': 25,
        'max_comm_cost': 100,
        'max_hubs': 5,
    }

    ### given parameter code ends here
    ### new parameter code ends here

    hub_placement_problem = OptimizedHubPlacement(parameters, seed=seed)
    instance = hub_placement_problem.get_instance()
    solve_status, solve_time = hub_placement_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")