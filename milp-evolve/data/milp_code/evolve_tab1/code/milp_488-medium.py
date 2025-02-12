import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.num_facilities > 0 and self.num_clients > 0
        assert self.min_opening_cost >= 0 and self.max_opening_cost >= self.min_opening_cost
        assert self.min_assignment_cost >= 0 and self.max_assignment_cost >= self.min_assignment_cost

        facilities = np.random.rand(self.num_facilities, 2)  # x, y coordinates
        clients = np.random.rand(self.num_clients, 2)  # x, y coordinates

        opening_costs = self.min_opening_cost + (self.max_opening_cost - self.min_opening_cost) * np.random.rand(self.num_facilities)
        assignment_costs = np.zeros((self.num_clients, self.num_facilities))

        for i in range(self.num_clients):
            for j in range(self.num_facilities):
                assignment_costs[i][j] = self.min_assignment_cost + (self.max_assignment_cost - self.min_assignment_cost) * np.linalg.norm(clients[i] - facilities[j])

        capacities = np.random.randint(self.min_capacity, self.max_capacity, size=self.num_facilities)
        demands = np.random.randint(self.min_demand, self.max_demand, size=self.num_clients)
        
        return {
            "facilities": facilities,
            "clients": clients,
            "opening_costs": opening_costs,
            "assignment_costs": assignment_costs,
            "capacities": capacities,
            "demands": demands
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        opening_costs = instance['opening_costs']
        assignment_costs = instance['assignment_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        
        model = Model("FacilityLocation")

        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(len(opening_costs))}
        assignments = {(i, j): model.addVar(vtype="B", name=f"Assign_{i}_{j}") for i in range(len(demands)) for j in range(len(opening_costs))}
        
        # Objective: minimize the total cost (opening + assignment)
        opening_cost_term = quicksum(opening_costs[j] * open_facilities[j] for j in range(len(opening_costs)))
        assignment_cost_term = quicksum(assignment_costs[i][j] * assignments[i, j] for i in range(len(demands)) for j in range(len(opening_costs)))
        model.setObjective(opening_cost_term + assignment_cost_term, "minimize")
        
        # Constraint: Each client must be assigned to exactly one facility
        for i in range(len(demands)):
            model.addCons(quicksum(assignments[i, j] for j in range(len(opening_costs))) == 1, f"ClientAssignment_{i}")

        # Constraint: A client can only be assigned to an opened facility
        for j in range(len(opening_costs)):
            for i in range(len(demands)):
                model.addCons(assignments[i, j] <= open_facilities[j], f"FacilityActivation_{i}_{j}")

        # Constraint: Facility capacities
        for j in range(len(opening_costs)):
            model.addCons(quicksum(demands[i] * assignments[i, j] for i in range(len(demands))) <= capacities[j], f"Capacity_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_facilities': 90,
        'num_clients': 50,
        'min_opening_cost': 800,
        'max_opening_cost': 2000,
        'min_assignment_cost': 3,
        'max_assignment_cost': 80,
        'min_capacity': 70,
        'max_capacity': 400,
        'min_demand': 4,
        'max_demand': 50,
    }

    fl = FacilityLocation(parameters, seed=42)
    instance = fl.generate_instance()
    solve_status, solve_time = fl.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")