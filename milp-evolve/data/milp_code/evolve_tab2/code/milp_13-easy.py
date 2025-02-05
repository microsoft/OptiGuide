import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class VesselPortAssignment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        """generate example data set"""
        vessel_processing_times = {}
        port_handling_costs = {}
        vessel_demands = {}
        berthing_capacity = {}
        port_overtime_cost = {}

        for v in range(1, self.num_vessels + 1):
            vessel_processing_times[v] = random.randint(2, 8)  

        for p in range(1, self.num_ports + 1):
            port_handling_costs[p] = random.randint(50, 200)
            berthing_capacity[p] = random.randint(2, 5)
            port_overtime_cost[p] = random.randint(100, 300)  
            for v in range(1, self.num_vessels + 1):
                vessel_demands[v, p] = random.randint(1, 3)

        res = {
            'vessel_processing_times': vessel_processing_times,
            'port_handling_costs': port_handling_costs,
            'vessel_demands': vessel_demands,
            'berthing_capacity': berthing_capacity,
            'port_overtime_cost': port_overtime_cost
        }

        return res

    def solve(self, instance):

        vessel_processing_times = instance['vessel_processing_times']
        port_handling_costs = instance['port_handling_costs']
        vessel_demands = instance['vessel_demands']
        berthing_capacity = instance['berthing_capacity']
        port_overtime_cost = instance['port_overtime_cost']

        model = Model("Vessel Port Assignment")

        # Variables
        x = {}  
        for v in range(1, self.num_vessels + 1):
            for p in range(1, self.num_ports + 1):
                x[v, p] = model.addVar(vtype="B", name=f"assign_v{v}_p{p}")

        overtime = {}
        for p in range(1, self.num_ports + 1):
            overtime[p] = model.addVar(vtype="C", name=f"overtime_p{p}")

        # Constraints
        for v in range(1, self.num_vessels + 1):
            model.addCons(quicksum(x[v, p] for p in range(1, self.num_ports + 1)) == 1, 
                          f"VesselAssignment_{v}")

        for p in range(1, self.num_ports + 1):
            model.addCons(quicksum(vessel_processing_times[v] * x[v, p] for v in range(1, self.num_vessels + 1)) 
                          <= berthing_capacity[p] + overtime[p], f"PortCapacity_{p}")

        # Objective function
        objective_expr = quicksum(port_handling_costs[p] * quicksum(x[v, p] for v in range(1, self.num_vessels + 1)) +
                                  port_overtime_cost[p] * overtime[p] 
                                  for p in range(1, self.num_ports + 1))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == "__main__":
    parameters = {
        'num_vessels': 300,
        'num_ports': 35,
        'port_capacity_factor': 9.0,
    }

    model = VesselPortAssignment(parameters)
    instance = model.generate_instance()
    solve_status, solve_time = model.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")