import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class EVFleetChargingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_vehicles * self.n_stations * self.density)
        
        # Vehicle power requirements and station power capacities
        vehicle_power_req = np.random.randint(self.min_power_req, self.max_power_req, size=self.n_vehicles)
        station_power_cap = np.random.randint(self.min_power_cap, self.max_power_cap, size=self.n_stations)
        
        # Charging efficiency and cost
        charging_efficiency = np.random.uniform(self.min_efficiency, self.max_efficiency, size=(self.n_stations, self.n_vehicles))
        station_activation_cost = np.random.randint(self.activation_cost_low, self.activation_cost_high, size=self.n_stations)
        
        res = {
            'vehicle_power_req': vehicle_power_req,
            'station_power_cap': station_power_cap,
            'charging_efficiency': charging_efficiency,
            'station_activation_cost': station_activation_cost,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        vehicle_power_req = instance['vehicle_power_req']
        station_power_cap = instance['station_power_cap']
        charging_efficiency = instance['charging_efficiency']
        station_activation_cost = instance['station_activation_cost']

        model = Model("EVFleetChargingOptimization")
        station_vars = {}
        allocation_vars = {}
        charge_level_vars = {}

        # Create variables and set objectives
        for j in range(self.n_stations):
            station_vars[j] = model.addVar(vtype="B", name=f"Station_{j}", obj=station_activation_cost[j])

        for v in range(self.n_vehicles):
            for s in range(self.n_stations):
                allocation_vars[(v, s)] = model.addVar(vtype="B", name=f"Vehicle_{v}_Station_{s}", obj=charging_efficiency[s][v] * vehicle_power_req[v])

        for s in range(self.n_stations):
            charge_level_vars[s] = model.addVar(vtype="C", name=f"Charge_Level_{s}")

        # Ensure each vehicle is assigned to one station
        for v in range(self.n_vehicles):
            model.addCons(quicksum(allocation_vars[(v, s)] for s in range(self.n_stations)) == 1, f"Vehicle_{v}_Assignment")

        # Power capacity constraints for stations
        for s in range(self.n_stations):
            model.addCons(quicksum(allocation_vars[(v, s)] * vehicle_power_req[v] for v in range(self.n_vehicles)) <= station_power_cap[s], f"Station_{s}_Capacity")
            model.addCons(charge_level_vars[s] == quicksum(allocation_vars[(v, s)] * vehicle_power_req[v] for v in range(self.n_vehicles)), f"Charge_Level_{s}")
            model.addCons(charge_level_vars[s] <= station_power_cap[s] * station_vars[s], f"Station_{s}_Active_Capacity")

        # Objective: Minimize total cost including energy usage and station activation costs
        objective_expr = quicksum(station_activation_cost[j] * station_vars[j] for j in range(self.n_stations)) + \
                         quicksum(charging_efficiency[s][v] * allocation_vars[(v, s)] * vehicle_power_req[v] for s in range(self.n_stations) for v in range(self.n_vehicles))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vehicles': 100,
        'n_stations': 50,
        'density': 0.79,
        'min_power_req': 45,
        'max_power_req': 200,
        'min_power_cap': 700,
        'max_power_cap': 1000,
        'min_efficiency': 0.8,
        'max_efficiency': 2.0,
        'activation_cost_low': 100,
        'activation_cost_high': 1000,
    }

    problem = EVFleetChargingOptimization(parameters, seed=seed)
    instance = problem.generate_instance()
    solve_status, solve_time = problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")