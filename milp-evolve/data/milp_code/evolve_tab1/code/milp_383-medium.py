import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EnhancedPublicTransportOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_buses = random.randint(self.min_buses, self.max_buses)
        n_stops = random.randint(self.min_stops, self.max_stops)
        n_demographic_groups = random.randint(self.min_demographic_groups, self.max_demographic_groups)

        # Costs and demands
        bus_costs = np.random.randint(50, 200, size=n_buses)
        stop_costs = np.random.randint(10, 100, size=n_stops)
        demographic_weights = np.random.normal(loc=1.0, scale=0.2, size=n_demographic_groups)
        peak_hours = np.random.choice([0, 1], size=(n_buses, 24))  # 24 hours in a day

        # Accessibility and distance matrices
        wheelchair_access = np.random.choice([0, 1], size=n_buses)
        walking_distances = np.random.randint(1, 10, size=(n_demographic_groups, n_stops))

        res = {
            'n_buses': n_buses,
            'n_stops': n_stops,
            'n_demographic_groups': n_demographic_groups,
            'bus_costs': bus_costs,
            'stop_costs': stop_costs,
            'demographic_weights': demographic_weights,
            'peak_hours': peak_hours,
            'wheelchair_access': wheelchair_access,
            'walking_distances': walking_distances
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_buses = instance['n_buses']
        n_stops = instance['n_stops']
        n_demographic_groups = instance['n_demographic_groups']
        bus_costs = instance['bus_costs']
        stop_costs = instance['stop_costs']
        demographic_weights = instance['demographic_weights']
        peak_hours = instance['peak_hours']
        wheelchair_access = instance['wheelchair_access']
        walking_distances = instance['walking_distances']

        model = Model("EnhancedPublicTransportOptimization")

        # Variables
        x = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(n_buses) for j in range(n_stops)}  # Bus i covers stop j 
        y = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(n_stops)}  # Stop j is activated

        # Objective function: Minimize total cost and total weighted walking distance
        total_cost = quicksum(bus_costs[i] for i in range(n_buses) if any(x[i, j] for j in range(n_stops))) + \
                     quicksum(stop_costs[j] * y[j] for j in range(n_stops)) + \
                     quicksum(walking_distances[k, j] * demographic_weights[k] * y[j] for k in range(n_demographic_groups) for j in range(n_stops))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(n_buses):
            model.addCons(quicksum(x[i, j] for j in range(n_stops)) == 1, name=f"bus_coverage_{i}")

        for j in range(n_stops):
            model.addCons(quicksum(x[i, j] for i in range(n_buses)) >= 1, name=f"stop_coverage_{j}")

        for i in range(n_buses):
            for j in range(n_stops):
                model.addCons(x[i, j] <= y[j], name=f"bus_stop_activation_{i}_{j}")

        # Ensure wheelchair accessibility for specific demographic groups
        for i in range(n_buses):
            if wheelchair_access[i] == 1:
                model.addCons(quicksum(x[i, j] for j in range(n_stops)) >= 1, name=f"wheelchair_access_{i}")

        # Ensure peak hours coverage
        for hour in range(24):
            if peak_hours[i, hour] == 1:
                model.addCons(quicksum(x[i, j] for i in range(n_buses) for j in range(n_stops)) >= 1, name=f"peak_hour_coverage_{hour}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_buses': 50,
        'max_buses': 300,
        'min_stops': 60,
        'max_stops': 200,
        'min_demographic_groups': 60,
        'max_demographic_groups': 200,
    }

    transportation_optimization = EnhancedPublicTransportOptimization(parameters, seed=seed)
    instance = transportation_optimization.generate_instance()
    solve_status, solve_time = transportation_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")