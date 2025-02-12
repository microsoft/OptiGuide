import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_facilities = random.randint(self.min_facilities, self.max_facilities)
        n_clients = random.randint(self.min_clients, self.max_clients)
        
        # Costs and capacities
        facility_setup_cost = np.random.randint(100, 500, size=n_facilities)
        transport_cost = np.random.randint(10, 100, size=(n_clients, n_facilities))
        facility_capacity = np.random.randint(50, 300, size=n_facilities)
        client_demand = np.random.randint(10, 40, size=n_clients)
        
        # Adjacency matrix
        adjacency_matrix = np.random.randint(0, 2, size=(n_clients, n_facilities))

        res = {
            'n_facilities': n_facilities,
            'n_clients': n_clients,
            'facility_setup_cost': facility_setup_cost,
            'transport_cost': transport_cost,
            'facility_capacity': facility_capacity,
            'client_demand': client_demand,
            'adjacency_matrix': adjacency_matrix,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_facilities = instance['n_facilities']
        n_clients = instance['n_clients']
        facility_setup_cost = instance['facility_setup_cost']
        transport_cost = instance['transport_cost']
        facility_capacity = instance['facility_capacity']
        client_demand = instance['client_demand']
        adjacency_matrix = instance['adjacency_matrix']
        
        model = Model("FacilityLocationOptimization")
        
        # Variables
        facility_open = {j: model.addVar(vtype="B", name=f"facility_open_{j}") for j in range(n_facilities)}
        client_assignment = {}
        for i in range(n_clients):
            for j in range(n_facilities):
                client_assignment[i, j] = model.addVar(vtype="B", name=f"client_assignment_{i}_{j}")

        # Objective function: Minimize total setup and transportation costs
        total_cost = quicksum(facility_open[j] * facility_setup_cost[j] for j in range(n_facilities)) + \
                     quicksum(client_assignment[i, j] * transport_cost[i, j] for i in range(n_clients) for j in range(n_facilities))
        model.setObjective(total_cost, "minimize")

        # Constraints
        # Each client must be assigned to exactly one facility
        for i in range(n_clients):
            model.addCons(quicksum(client_assignment[i, j] for j in range(n_facilities)) == 1, name=f"client_assignment_{i}")
        
        # Facility capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(client_assignment[i, j] * client_demand[i] for i in range(n_clients)) <= facility_capacity[j] * facility_open[j],
                          name=f"facility_capacity_{j}")
        
        # Clients can only be assigned to open facilities
        for i in range(n_clients):
            for j in range(n_facilities):
                model.addCons(client_assignment[i, j] <= facility_open[j], name=f"only_open_facilities_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 123
    parameters = {
        'min_facilities': 15,
        'max_facilities': 120,
        'min_clients': 30,
        'max_clients': 200,
    }
    
    facility_location_optimizer = FacilityLocationOptimization(parameters, seed=seed)
    instance = facility_location_optimizer.generate_instance()
    solve_status, solve_time = facility_location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")