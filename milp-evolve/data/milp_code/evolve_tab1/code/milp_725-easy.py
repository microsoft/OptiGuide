import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SustainableTransportOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_modes = random.randint(self.min_modes, self.max_modes)
        num_routes = random.randint(self.min_routes, self.max_routes)

        # Cost matrices
        transport_cost = np.random.randint(20, 200, size=(num_routes, num_modes))
        environmental_impact = np.random.randint(50, 300, size=(num_routes, num_modes))

        # Route demand
        route_demand = np.random.randint(100, 500, size=num_routes)

        # Mode capacities
        mode_capacity = np.random.randint(5000, 10000, size=num_modes)

        route_capacity = np.random.randint(1000, 8000, size=num_routes)

        res = {
            'num_modes': num_modes,
            'num_routes': num_routes,
            'transport_cost': transport_cost,
            'environmental_impact': environmental_impact,
            'route_demand': route_demand,
            'mode_capacity': mode_capacity,
            'route_capacity': route_capacity,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_modes = instance['num_modes']
        num_routes = instance['num_routes']
        transport_cost = instance['transport_cost']
        environmental_impact = instance['environmental_impact']
        route_demand = instance['route_demand']
        mode_capacity = instance['mode_capacity']
        route_capacity = instance['route_capacity']

        model = Model("SustainableTransportOptimization")

        # Variables
        NVariable_transport_mode = {(i, j): model.addVar(vtype="B", name=f"NVariable_transport_mode_{i}_{j}") for i in range(num_routes) for j in range(num_modes)}
        AVariable_amount_transported = {(i, j): model.addVar(vtype="C", name=f"AVariable_amount_transported_{i}_{j}") for i in range(num_routes) for j in range(num_modes)}

        # Objective function: Minimize total costs including environmental impact
        AObjective_total_cost = quicksum(NVariable_transport_mode[i, j] * (transport_cost[i, j] + environmental_impact[i, j]) for i in range(num_routes) for j in range(num_modes))

        model.setObjective(AObjective_total_cost, "minimize")

        # Constraints
        for i in range(num_routes):
            model.addCons(quicksum(AVariable_amount_transported[i, j] for j in range(num_modes)) == route_demand[i], name=f"TConstraint_route_demand_{i}")

        for j in range(num_modes):
            model.addCons(quicksum(AVariable_amount_transported[i, j] for i in range(num_routes)) <= mode_capacity[j], name=f"TConstraint_mode_capacity_{j}")

        for i in range(num_routes):
            model.addCons(quicksum(NVariable_transport_mode[i, j] * route_capacity[i] for j in range(num_modes)) >= route_demand[i], name=f"TConstraint_route_capacity_{i}")

            for j in range(num_modes):
                model.addCons(AVariable_amount_transported[i, j] <= route_demand[i] * NVariable_transport_mode[i, j], name=f"TConstraint_route_mode_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_modes': 18,
        'max_modes': 30,
        'min_routes': 500,
        'max_routes': 700,
    }

    optimization = SustainableTransportOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")