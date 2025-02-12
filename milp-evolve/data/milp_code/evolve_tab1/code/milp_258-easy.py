import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EVChargingStationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data Generation
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def unit_transportation_costs(self):
        return np.random.rand(self.n_customers, self.n_stations) * self.transport_cost_scale

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        station_capacities = self.randint(self.n_stations, self.station_capacity_interval)
        fixed_costs = self.randint(self.n_stations, self.fixed_cost_interval)
        transport_costs = self.unit_transportation_costs()

        res = {
            'demands': demands,
            'station_capacities': station_capacities,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs
        }

        return res

    # MILP Solver
    def solve(self, instance):
        demands = instance['demands']
        station_capacities = instance['station_capacities']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']

        n_customers = len(demands)
        n_stations = len(station_capacities)

        model = Model("EVChargingStationOptimization")

        # Decision variables
        open_stations = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_stations)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(n_customers) for j in range(n_stations)}

        # Objective: Minimize cost function
        objective_expr = quicksum(self.cost_weight * fixed_costs[j] * open_stations[j] for j in range(n_stations)) + \
                         quicksum(self.transport_cost_weight * transport_costs[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_stations))

        model.setObjective(objective_expr, "minimize")

        # Demand satisfaction constraints
        for i in range(n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(n_stations)) == demands[i], f"Demand_{i}")

        # Station capacity constraints
        for j in range(n_stations):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) <= station_capacities[j] * open_stations[j], f"StationCapacity_{j}")

        # Financial resource limitations constraint
        model.addCons(quicksum(fixed_costs[j] * open_stations[j] for j in range(n_stations)) + \
                      quicksum(transport_costs[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_stations)) <= self.budget, "BudgetConstraint")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 90,
        'n_stations': 200,
        'demand_interval': (400, 2000),
        'station_capacity_interval': (500, 2000),
        'fixed_cost_interval': (500, 2500),
        'transport_cost_scale': 400.0,
        'cost_weight': 7.0,
        'transport_cost_weight': 3.0,
        'budget': 500000,
    }

    ev_charging_optimization = EVChargingStationOptimization(parameters, seed=seed)
    instance = ev_charging_optimization.generate_instance()
    solve_status, solve_time = ev_charging_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")