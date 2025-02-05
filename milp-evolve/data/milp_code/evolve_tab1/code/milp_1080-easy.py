import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # number of facilities and clients
        n_facilities = self.n_facilities
        n_clients = self.n_clients

        # opening costs for each facility
        open_cost = np.random.randint(self.max_open_cost, size=n_facilities) + 1

        # service cost between each facility and each client
        service_cost = np.random.randint(self.max_service_cost, size=(n_facilities, n_clients)) + 1

        res = {
            'open_cost': open_cost,
            'service_cost': service_cost
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        open_cost = instance['open_cost']
        service_cost = instance['service_cost']

        model = Model("FacilityLocation")
        
        # Create variables
        y = {}
        x = {}
        for i in range(self.n_facilities):
            y[i] = model.addVar(vtype="B", name=f"y_{i}", obj=open_cost[i])
        for i in range(self.n_facilities):
            for j in range(self.n_clients):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}", obj=service_cost[i, j])

        # Each client must be served by exactly one facility
        for j in range(self.n_clients):
            model.addCons(quicksum(x[i, j] for i in range(self.n_facilities)) == 1, f"serve_{j}")

        # A client can only be assigned to an open facility
        for i in range(self.n_facilities):
            for j in range(self.n_clients):
                model.addCons(x[i, j] <= y[i], f"assign_{i}_{j}")

        # Objective: Minimize total opening and service costs
        model.setMinimize()

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 30,
        'n_clients': 450,
        'max_open_cost': 300,
        'max_service_cost': 100,
    }

    facility_location_problem = FacilityLocation(parameters, seed=seed)
    instance = facility_location_problem.generate_instance()
    solve_status, solve_time = facility_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")