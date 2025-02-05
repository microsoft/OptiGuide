import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyServiceDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################

    def generate_instance(self):
        # Setup costs for EMS stations
        setup_cost = np.random.randint(self.min_setup_cost, self.max_setup_cost, self.number_of_stations)

        # Assignment costs of ambulances to stations
        assignment_costs = np.random.randint(self.min_assignment_cost, self.max_assignment_cost, (self.number_of_stations, self.number_of_ambulances))

        # Shift costs for each ambulance
        shift_costs = np.random.randint(self.min_shift_cost, self.max_shift_cost, (self.number_of_ambulances, self.number_of_shifts))

        # Capacity of EMS stations
        station_capacities = np.random.randint(self.min_station_capacity, self.max_station_capacity, self.number_of_stations)

        # Demand points representing emergency cases
        demand_points = np.random.randint(0, self.number_of_emergency_cases, self.number_of_demand_points)

        distances = np.random.rand(self.number_of_stations, self.number_of_demand_points)
        coverage_feasibility = np.where(distances <= self.max_coverage_distance, 1, 0)

        res = {
            'setup_cost': setup_cost,
            'assignment_costs': assignment_costs,
            'shift_costs': shift_costs,
            'station_capacities': station_capacities,
            'demand_points': demand_points,
            'coverage_feasibility': coverage_feasibility,
        }
        return res

    ################# PySCIPOpt Modeling #################

    def solve(self, instance):
        setup_cost = instance['setup_cost']
        assignment_costs = instance['assignment_costs']
        shift_costs = instance['shift_costs']
        station_capacities = instance['station_capacities']
        demand_points = instance['demand_points']
        coverage_feasibility = instance['coverage_feasibility']

        number_of_stations = len(setup_cost)
        number_of_ambulances = len(assignment_costs[0])
        number_of_shifts = len(shift_costs[0])
        number_of_demand_points = len(demand_points)

        model = Model("EmergencyServiceDeployment")

        # Decision variables
        station_setup = {i: model.addVar(vtype="B", name=f"station_setup_{i}") for i in range(number_of_stations)}
        ambulance_assignment = {(i, j): model.addVar(vtype="B", name=f"ambulance_assignment_{i}_{j}") for i in range(number_of_stations) for j in range(number_of_ambulances)}
        shift_assignment = {(j, s): model.addVar(vtype="B", name=f"shift_assignment_{j}_{s}") for j in range(number_of_ambulances) for s in range(number_of_shifts)}

        # Objective: Minimize total cost (setup costs + assignment costs + shift costs)
        objective_expr = quicksum(setup_cost[i] * station_setup[i] for i in range(number_of_stations))
        objective_expr += quicksum(assignment_costs[i][j] * ambulance_assignment[(i, j)] for i in range(number_of_stations) for j in range(number_of_ambulances))
        objective_expr += quicksum(shift_costs[j][s] * shift_assignment[(j, s)] for j in range(number_of_ambulances) for s in range(number_of_shifts))

        model.setObjective(objective_expr, "minimize")

        # Constraint: Each emergency demand point must be covered by at least one station within coverage distance
        for k in range(number_of_demand_points):
            model.addCons(quicksum(station_setup[i] * coverage_feasibility[i][k] for i in range(number_of_stations)) >= 1, f"DemandCoverage_{k}")

        # Constraint: Ambulance station capacity constraints
        for i in range(number_of_stations):
            model.addCons(quicksum(ambulance_assignment[(i, j)] for j in range(number_of_ambulances)) <= station_capacities[i], f"StationCapacity_{i}")

        # Constraint: Ensure fairness of ambulance distribution
        for j in range(number_of_ambulances):
            model.addCons(quicksum(ambulance_assignment[(i, j)] for i in range(number_of_stations)) <= 1, f"AmbulanceFairness_{j}")

        # Constraint: Ensure each ambulance follows shift regulations
        for j in range(number_of_ambulances):
            model.addCons(quicksum(shift_assignment[(j, s)] for s in range(number_of_shifts)) <= self.max_shifts_per_ambulance, f"ShiftRegulation_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_stations': 400,
        'number_of_ambulances': 180,
        'number_of_shifts': 12,
        'number_of_demand_points': 500,
        'number_of_emergency_cases': 1000,
        'min_setup_cost': 5000,
        'max_setup_cost': 10000,
        'min_assignment_cost': 600,
        'max_assignment_cost': 3000,
        'min_shift_cost': 700,
        'max_shift_cost': 2500,
        'min_station_capacity': 6,
        'max_station_capacity': 100,
        'max_coverage_distance': 0.74,
        'max_shifts_per_ambulance': 100,
    }

    emergency_service = EmergencyServiceDeployment(parameters, seed=seed)
    instance = emergency_service.generate_instance()
    solve_status, solve_time = emergency_service.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")