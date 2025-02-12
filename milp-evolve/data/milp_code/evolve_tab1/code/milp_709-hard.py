import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class SchoolBusRoutingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_buses = random.randint(self.min_buses, self.max_buses)
        num_students = random.randint(self.min_students, self.max_students)

        # Cost matrices
        transportation_costs = np.random.randint(10, 100, size=(num_students, num_buses))
        fixed_costs = np.random.randint(500, 2000, size=num_buses)

        # Student counts per bus (bus capacities)
        bus_capacities = np.random.randint(10, 50, size=num_buses)

        res = {
            'num_buses': num_buses,
            'num_students': num_students,
            'transportation_costs': transportation_costs,
            'fixed_costs': fixed_costs,
            'bus_capacities': bus_capacities,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_buses = instance['num_buses']
        num_students = instance['num_students']
        transportation_costs = instance['transportation_costs']
        fixed_costs = instance['fixed_costs']
        bus_capacities = instance['bus_capacities']

        model = Model("SchoolBusRoutingOptimization")

        # Variables
        bus = {j: model.addVar(vtype="B", name=f"bus_{j}") for j in range(num_buses)}
        assignment = {(i, j): model.addVar(vtype="B", name=f"assignment_{i}_{j}") for i in range(num_students) for j in range(num_buses)}

        # Objective function: Minimize total costs
        total_cost = quicksum(assignment[i, j] * transportation_costs[i, j] for i in range(num_students) for j in range(num_buses)) + \
                     quicksum(bus[j] * fixed_costs[j] for j in range(num_buses))

        model.setObjective(total_cost, "minimize")

        # Constraints
        # Each student must be assigned to exactly one bus
        for i in range(num_students):
            model.addCons(quicksum(assignment[i, j] for j in range(num_buses)) == 1, name=f"student_assignment_{i}")

        # Logical constraints: A student can only be assigned to a bus if the bus is operational
        for j in range(num_buses):
            for i in range(num_students):
                model.addCons(assignment[i, j] <= bus[j], name=f"bus_operation_constraint_{i}_{j}")

        # Capacity constraints: A bus can only have up to its capacity
        for j in range(num_buses):
            model.addCons(quicksum(assignment[i, j] for i in range(num_students)) <= bus_capacities[j], name=f"bus_capacity_{j}")

        model.optimize()

        solution = {"assignments": {}, "buses": {}}
        if model.getStatus() == "optimal":
            for j in range(num_buses):
                solution["buses"][j] = model.getVal(bus[j])
            for i in range(num_students):
                for j in range(num_buses):
                    if model.getVal(assignment[i, j]) > 0.5:
                        solution["assignments"][i] = j

        return solution, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_buses': 50,
        'max_buses': 80,
        'min_students': 30,
        'max_students': 800,
    }

    optimization = SchoolBusRoutingOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solution, total_cost = optimization.solve(instance)

    print(f"Solution: {solution}")
    print(f"Total Cost: {total_cost:.2f}")