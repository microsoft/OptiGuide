import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HarborHighwayOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_harbors = random.randint(self.min_harbors, self.max_harbors)
        num_cities = random.randint(self.min_cities, self.max_cities)

        # Cost matrices
        city_connection_costs = np.random.randint(50, 300, size=(num_cities, num_harbors))
        operational_costs_harbor = np.random.randint(1000, 3000, size=num_harbors)
        highway_construction_costs = np.random.randint(500, 1500, size=(num_cities, num_cities))

        # City demands
        city_population = np.random.randint(100, 500, size=num_cities)

        # Highway capacities
        highway_capacity = np.random.randint(200, 1000, size=(num_cities, num_cities))

        res = {
            'num_harbors': num_harbors,
            'num_cities': num_cities,
            'city_connection_costs': city_connection_costs,
            'operational_costs_harbor': operational_costs_harbor,
            'highway_construction_costs': highway_construction_costs,
            'city_population': city_population,
            'highway_capacity': highway_capacity,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_harbors = instance['num_harbors']
        num_cities = instance['num_cities']
        city_connection_costs = instance['city_connection_costs']
        operational_costs_harbor = instance['operational_costs_harbor']
        highway_construction_costs = instance['highway_construction_costs']
        city_population = instance['city_population']
        highway_capacity = instance['highway_capacity']

        model = Model("HarborHighwayOptimization")

        # Variables
        harbor_opened = {j: model.addVar(vtype="B", name=f"harbor_opened_{j}") for j in range(num_harbors)}
        highway_construction = {(i, j): model.addVar(vtype="B", name=f"highway_construction_{i}_{j}") for i in range(num_cities) for j in range(num_cities)}
        city_connection = {(i, j): model.addVar(vtype="B", name=f"city_connection_{i}_{j}") for i in range(num_cities) for j in range(num_harbors)}
        number_of_routes = {(i, j): model.addVar(vtype="I", name=f"number_of_routes_{i}_{j}") for i in range(num_cities) for j in range(num_harbors)}

        # Objective function: Minimize total costs
        total_cost = quicksum(city_connection[i, j] * city_connection_costs[i, j] for i in range(num_cities) for j in range(num_harbors)) + \
                     quicksum(harbor_opened[j] * operational_costs_harbor[j] for j in range(num_harbors)) + \
                     quicksum(highway_construction[i, j] * highway_construction_costs[i, j] for i in range(num_cities) for j in range(num_cities)) + \
                     quicksum(highway_capacity[i, j] * highway_construction[i, j] for i in range(num_cities) for j in range(num_cities))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_cities):
            model.addCons(quicksum(city_connection[i, j] for j in range(num_harbors)) >= 1, name=f"city_connection_{i}")

        for i in range(num_cities):
            for j in range(num_harbors):
                model.addCons(city_connection[i, j] <= harbor_opened[j], name=f"harbor_city_{i}_{j}")

        for i in range(num_cities):
            model.addCons(quicksum(number_of_routes[i, j] for j in range(num_harbors)) <= city_population[i], name=f"capacity_{i}")

        for i in range(num_cities):
            for j in range(num_harbors):
                model.addCons(number_of_routes[i, j] == city_population[i] * city_connection[i, j], name=f"routes_city_population_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_harbors': 10,
        'max_harbors': 100,
        'min_cities': 40,
        'max_cities': 300,
    }

    optimization = HarborHighwayOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")