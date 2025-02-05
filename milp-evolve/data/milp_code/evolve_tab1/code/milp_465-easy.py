import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HealthcareResourceNetworkFlow:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_resources = random.randint(self.min_resources, self.max_resources)
        n_facilities = random.randint(self.min_facilities, self.max_facilities)

        # Costs and capacities
        resource_costs = np.random.randint(10, 100, size=(n_resources, n_facilities))
        personnel_costs = np.random.randint(20, 200, size=n_facilities)
        facility_capacity = np.random.randint(50, 200, size=n_facilities)
        resource_supply = np.random.randint(5, 20, size=n_resources)

        res = {
            'n_resources': n_resources,
            'n_facilities': n_facilities,
            'resource_costs': resource_costs,
            'personnel_costs': personnel_costs,
            'facility_capacity': facility_capacity,
            'resource_supply': resource_supply
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_resources = instance['n_resources']
        n_facilities = instance['n_facilities']
        resource_costs = instance['resource_costs']
        personnel_costs = instance['personnel_costs']
        facility_capacity = instance['facility_capacity']
        resource_supply = instance['resource_supply']

        model = Model("HealthcareResourceNetworkFlow")

        # Variables
        flow = {}
        for i in range(n_resources):
            for j in range(n_facilities):
                flow[i, j] = model.addVar(vtype="C", name=f"flow_{i}_{j}")

        c = {j: model.addVar(vtype="B", name=f"c_{j}") for j in range(n_facilities)}

        # Objective function: Minimize total cost (resource allocation + personnel)
        total_cost = quicksum(flow[i, j] * resource_costs[i, j] for i in range(n_resources) for j in range(n_facilities)) + \
                     quicksum(c[j] * personnel_costs[j] for j in range(n_facilities))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_resources):
            model.addCons(quicksum(flow[i, j] for j in range(n_facilities)) == resource_supply[i], name=f"resource_supply_{i}")

        for j in range(n_facilities):
            model.addCons(quicksum(flow[i, j] for i in range(n_resources)) <= facility_capacity[j] * c[j], name=f"facility_capacity_{j}")
            model.addCons(quicksum(flow[i, j] for i in range(n_resources)) >= 0 * c[j], name=f"facility_activation_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_resources': 180,
        'max_resources': 336,
        'min_facilities': 48,
        'max_facilities': 900,
        'min_resources_per_facility': 25,
    }

    resource_network_flow = HealthcareResourceNetworkFlow(parameters, seed=seed)
    instance = resource_network_flow.generate_instance()
    solve_status, solve_time = resource_network_flow.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")