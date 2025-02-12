import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocationTransportation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        # Randomly generate fixed costs for opening a facility
        fixed_costs = np.random.randint(self.min_cost, self.max_cost, self.number_of_facilities)

        # Randomly generate transportation costs between facilities and nodes
        transportation_costs = np.random.randint(self.min_cost, self.max_cost, (self.number_of_facilities, self.number_of_nodes))

        # Randomly generate capacities of facilities
        facility_capacities = np.random.randint(self.min_cap, self.max_cap, self.number_of_facilities)

        # Randomly generate demands for nodes
        node_demands = np.random.randint(self.min_demand, self.max_demand, self.number_of_nodes)
        
        res = {
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'facility_capacities': facility_capacities,
            'node_demands': node_demands,
        }
        
        ### new instance data code ends here
        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        facility_capacities = instance['facility_capacities']
        node_demands = instance['node_demands']
        
        number_of_facilities = len(fixed_costs)
        number_of_nodes = len(node_demands)
        
        model = Model("FacilityLocationTransportation")
        open_facility = {}
        transport_goods = {}

        # Decision variables: y[j] = 1 if facility j is open
        for j in range(number_of_facilities):
            open_facility[j] = model.addVar(vtype="B", name=f"y_{j}")

        # Decision variables: x[i][j] = amount of goods transported from facility j to node i
        for i in range(number_of_nodes):
            for j in range(number_of_facilities):
                transport_goods[(i, j)] = model.addVar(vtype="C", name=f"x_{i}_{j}")

        # Objective: Minimize total cost
        objective_expr = quicksum(fixed_costs[j] * open_facility[j] for j in range(number_of_facilities)) + \
                         quicksum(transportation_costs[j][i] * transport_goods[(i, j)] for i in range(number_of_nodes) for j in range(number_of_facilities))
        model.setObjective(objective_expr, "minimize")

        # Constraints: Each node's demand must be met
        for i in range(number_of_nodes):
            model.addCons(
                quicksum(transport_goods[(i, j)] for j in range(number_of_facilities)) == node_demands[i],
                f"NodeDemand_{i}"
            )

        # Constraints: Facility capacity must not be exceeded 
        for j in range(number_of_facilities):
            model.addCons(
                quicksum(transport_goods[(i, j)] for i in range(number_of_nodes)) <= facility_capacities[j] * open_facility[j],
                f"FacilityCapacity_{j}"
            )

        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    parameters = {
        'number_of_facilities': 350,
        'number_of_nodes': 120,
        'min_cost': 60,
        'max_cost': 400,
        'min_cap': 150,
        'max_cap': 200,
        'min_demand': 120,
        'max_demand': 350,
    }
    seed = 42
    facility_location = FacilityLocationTransportation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")