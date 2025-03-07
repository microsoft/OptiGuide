import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FacilityLocationProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################

    def generate_instance(self):
        # Generating facility opening costs and capacities
        opening_costs = np.random.randint(self.min_opening_cost, self.max_opening_cost, self.number_of_facilities)
        capacities = np.random.randint(self.min_capacity, self.max_capacity, self.number_of_facilities)

        # Generating clients demand and assignment costs
        demands = np.random.randint(self.min_demand, self.max_demand, self.number_of_clients)
        assignment_costs = np.random.randint(self.min_assignment_cost, self.max_assignment_cost, (self.number_of_clients, self.number_of_facilities))

        # Generating transportation capacities and costs
        transport_capacity = np.random.randint(self.min_transport_capacity, self.max_transport_capacity, (self.number_of_facilities, self.number_of_facilities))
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost, (self.number_of_facilities, self.number_of_facilities))

        res = {
            'opening_costs': opening_costs,
            'capacities': capacities,
            'demands': demands,
            'assignment_costs': assignment_costs,
            'transport_capacity': transport_capacity,
            'transport_costs': transport_costs,
        }

        return res

    ################# PySCIPOpt Modeling #################

    def solve(self, instance):
        opening_costs = instance['opening_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        assignment_costs = instance['assignment_costs']
        transport_capacity = instance['transport_capacity']
        transport_costs = instance['transport_costs']

        number_of_facilities = len(opening_costs)
        number_of_clients = len(demands)

        model = Model("FacilityLocationProblem")

        # Decision variables
        open_facility = {i: model.addVar(vtype="B", name=f"open_facility_{i}") for i in range(number_of_facilities)}
        assign_client = {(c, f): model.addVar(vtype="B", name=f"assign_client_{c}_{f}") for c in range(number_of_clients) for f in range(number_of_facilities)}
        transport_quant = {(i, j): model.addVar(vtype="I", name=f"transport_quant_{i}_{j}") for i in range(number_of_facilities) for j in range(number_of_facilities)}
        
        # Introduce auxiliary variables for convex hull formulation
        Y = {(c, f): model.addVar(vtype="C", name=f"Y_{c}_{f}") for c in range(number_of_clients) for f in range(number_of_facilities)}

        # Objective: Minimize total cost (opening costs + assignment costs + transport costs)
        objective_expr = quicksum(opening_costs[f] * open_facility[f] for f in range(number_of_facilities))
        objective_expr += quicksum(assignment_costs[c][f] * assign_client[(c, f)] for c in range(number_of_clients) for f in range(number_of_facilities))
        objective_expr += quicksum(transport_costs[i][j] * transport_quant[(i, j)] for i in range(number_of_facilities) for j in range(number_of_facilities))

        model.setObjective(objective_expr, "minimize")

        # Constraint: Each client is assigned to exactly one facility
        for c in range(number_of_clients):
            model.addCons(quicksum(assign_client[(c, f)] for f in range(number_of_facilities)) == 1, f"ClientAssignment_{c}")

        # Convex Hull Capacity Constraints
        
        # Ensure Y_cf <= open_facility_f * bigM, bigM being capacity upper bound
        M = self.max_capacity
        for c in range(number_of_clients):
            for f in range(number_of_facilities):
                # Y_cf <= BigM * open_facility_f
                model.addCons(Y[(c, f)] <= M * open_facility[f], f"ConvexCapacity1_{c}_{f}")
                # assignment * demand == Y_cf
                model.addCons(Y[(c, f)] == demands[c] * assign_client[(c, f)], f"ConvexCapacity2_{c}_{f}")

        # New Capacity constraints
        for f in range(number_of_facilities):
            model.addCons(quicksum(Y[(c, f)] for c in range(number_of_clients)) <= capacities[f] * open_facility[f], f"FacilityCapacity_{f}")

        # Constraint: Transport capacity constraints
        for i in range(number_of_facilities):
            for j in range(number_of_facilities):
                model.addCons(transport_quant[(i, j)] <= transport_capacity[i][j], f"TransportCapacity_{i}_{j}")

        # Constraint: Ensure flow balance at each facility
        for f in range(number_of_facilities):
            incoming_flow = quicksum(transport_quant[(i, f)] for i in range(number_of_facilities))
            outgoing_flow = quicksum(transport_quant[(f, j)] for j in range(number_of_facilities))
            client_supply = quicksum(demands[c] * assign_client[(c, f)] for c in range(number_of_clients))
            model.addCons(incoming_flow + client_supply == outgoing_flow + capacities[f] * open_facility[f], f"FlowBalance_{f}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_facilities': 30,
        'number_of_clients': 150,
        'min_opening_cost': 100,
        'max_opening_cost': 1000,
        'min_capacity': 50,
        'max_capacity': 300,
        'min_demand': 1,
        'max_demand': 10,
        'min_assignment_cost': 10,
        'max_assignment_cost': 100,
        'min_transport_capacity': 50,
        'max_transport_capacity': 500,
        'min_transport_cost': 1,
        'max_transport_cost': 50,
    }

    facility_location = FacilityLocationProblem(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")