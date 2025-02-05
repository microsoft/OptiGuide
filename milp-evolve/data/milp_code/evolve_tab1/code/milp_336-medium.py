import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum


class MachineryResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        n_machinery = random.randint(self.min_machinery, self.max_machinery)
        n_fields = random.randint(self.min_fields, self.max_fields)
        n_weather = random.randint(self.min_weather, self.max_weather)
        n_crops = random.randint(2, 5)  # Different crop types

        # Cost matrices
        transport_costs = np.random.randint(50, 500, size=(n_machinery, n_fields))
        operation_costs = np.random.randint(200, 2000, size=n_machinery)
        harvesting_costs = np.random.randint(100, 1000, size=(n_machinery, n_weather))

        # Capacities and demands
        machinery_capacity = np.random.randint(100, 400, size=n_machinery)
        field_demand = np.random.randint(50, 300, size=(n_fields, n_weather, n_crops))

        res = {
            'n_machinery': n_machinery,
            'n_fields': n_fields,
            'n_weather': n_weather,
            'n_crops': n_crops,
            'transport_costs': transport_costs,
            'operation_costs': operation_costs,
            'harvesting_costs': harvesting_costs,
            'machinery_capacity': machinery_capacity,
            'field_demand': field_demand
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_machinery = instance['n_machinery']
        n_fields = instance['n_fields']
        n_weather = instance['n_weather']
        n_crops = instance['n_crops']
        transport_costs = instance['transport_costs']
        operation_costs = instance['operation_costs']
        harvesting_costs = instance['harvesting_costs']
        machinery_capacity = instance['machinery_capacity']
        field_demand = instance['field_demand']

        model = Model("MachineryResourceAllocation")

        # Variables
        machinery_allocate = {}
        for m in range(n_machinery):
            for f in range(n_fields):
                machinery_allocate[m, f] = model.addVar(vtype="B", name=f"machinery_allocate_{m}_{f}")

        machinery_active = {m: model.addVar(vtype="B", name=f"machinery_active_{m}") for m in range(n_machinery)}
        harvest_activate = {}
        for m in range(n_machinery):
            for w in range(n_weather):
                harvest_activate[m, w] = model.addVar(vtype="B", name=f"harvest_activate_{m}_{w}")

        # Objective function: Minimize total cost
        total_cost = quicksum(machinery_allocate[m, f] * transport_costs[m, f] for m in range(n_machinery) for f in range(n_fields)) + \
                     quicksum(machinery_active[m] * operation_costs[m] for m in range(n_machinery)) + \
                     quicksum(harvest_activate[m, w] * harvesting_costs[m, w] for m in range(n_machinery) for w in range(n_weather))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for f in range(n_fields):
            for w in range(n_weather):
                for c in range(n_crops):
                    model.addCons(quicksum(machinery_allocate[m, f] * machinery_capacity[m] * harvest_activate[m, w] for m in range(n_machinery)) >= field_demand[f, w, c], 
                                  name=f"field_demand_{f}_{w}_{c}")

        for m in range(n_machinery):
            for w in range(n_weather):
                model.addCons(harvest_activate[m, w] <= machinery_active[m], name=f"weather_operation_{m}_{w}")

            model.addCons(quicksum(machinery_allocate[m, f] for f in range(n_fields)) >= machinery_active[m], name=f"machinery_active_{m}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_machinery': 10,
        'max_machinery': 75,
        'min_fields': 30,
        'max_fields': 240,
        'min_weather': 24,
        'max_weather': 28,
    }

    allocation = MachineryResourceAllocation(parameters, seed=seed)
    instance = allocation.get_instance()
    solve_status, solve_time = allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")