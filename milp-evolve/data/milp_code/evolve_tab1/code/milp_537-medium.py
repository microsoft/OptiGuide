import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class WarehouseLocationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_warehouses = random.randint(self.min_warehouses, self.max_warehouses)
        n_locations = random.randint(self.min_locations, self.max_locations)

        # Cost matrices
        service_costs = np.random.randint(10, 100, size=(n_warehouses, n_locations))
        fixed_costs = np.random.randint(500, 1500, size=n_warehouses)

        # Energy consumption
        energy_consumption = np.random.rand(n_warehouses) * 10  # Consumption rate per vehicle
        initial_energy_levels = np.random.rand(n_warehouses) * 100  # Initial energy levels

        # Charging stations
        n_charging_stations = random.randint(self.min_charging_stations, self.max_charging_stations)
        charging_station_location = np.random.choice(n_locations, n_charging_stations, replace=False)

        ### New Data: Distances ###
        distances = np.random.rand(n_warehouses, n_locations) * 50  # Random distances in kilometers

        res = {
            'n_warehouses': n_warehouses,
            'n_locations': n_locations,
            'service_costs': service_costs,
            'fixed_costs': fixed_costs,
            'energy_consumption': energy_consumption,
            'initial_energy_levels': initial_energy_levels,
            'n_charging_stations': n_charging_stations,
            'charging_station_location': charging_station_location,
            'distances': distances,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_warehouses = instance['n_warehouses']
        n_locations = instance['n_locations']
        service_costs = instance['service_costs']
        fixed_costs = instance['fixed_costs']
        energy_consumption = instance['energy_consumption']
        initial_energy_levels = instance['initial_energy_levels']
        charging_station_location = instance['charging_station_location']
        n_charging_stations = instance['n_charging_stations']
        distances = instance['distances']

        model = Model("WarehouseLocationOptimization")

        # Variables
        y = {i: model.addVar(vtype="B", name=f"y_{i}") for i in range(n_warehouses)}
        x = {}
        for i in range(n_warehouses):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # New variables: Energy levels and charging status
        energy = {i: model.addVar(vtype="C", name=f"energy_{i}") for i in range(n_warehouses)}
        charging_status = {}
        for i in range(n_warehouses):
            for s in range(n_charging_stations):
                charging_status[i, s] = model.addVar(vtype="B", name=f"charging_status_{i}_{s}")

        ### New variables: alternative servicing ###
        alternative = {}
        for j in range(n_locations):
            alternative[j] = model.addVar(vtype="B", name=f"alternative_{j}")

        # Objective function: Minimize total cost and energy consumption
        total_cost = quicksum(x[i, j] * service_costs[i, j] for i in range(n_warehouses) for j in range(n_locations)) + \
                     quicksum(y[i] * fixed_costs[i] for i in range(n_warehouses)) + \
                     quicksum(energy[i] for i in range(n_warehouses))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for j in range(n_locations):
            model.addCons(quicksum(x[i,j] for i in range(n_warehouses)) == 1, name=f"location_coverage_{j}")

        # Logical constraints: A warehouse can only serve locations if it is open
        for i in range(n_warehouses):
            for j in range(n_locations):
                model.addCons(x[i,j] <= y[i], name=f"warehouse_to_location_{i}_{j}")

        # Energy constraints
        for i in range(n_warehouses):
            model.addCons(energy[i] == initial_energy_levels[i] - quicksum(x[i, j] * energy_consumption[i] for j in range(n_locations)), name=f"initial_energy_{i}")
            model.addCons(energy[i] >= 0, name=f"energy_non_negative_{i}")
            for s in range(n_charging_stations):
                model.addCons(energy[i] <= initial_energy_levels[i] + charging_status[i, s] * 100, name=f"charging_constraint_{i}_{s}")
                if charging_station_location[s] in range(n_locations):
                    model.addCons(charging_status[i, s] <= 1,  name=f"charging_status_constraint_{i}_{s}")

        # New logical constraints: alternative servicing and distance constraints
        big_M = 1e6
        for i in range(n_warehouses):
            for j in range(n_locations):
                model.addCons(x[i, j] * distances[i, j] <= big_M * (1 - alternative[j]), name=f"distance_servicing_{i}_{j}")
                model.addCons(alternative[j] <= 1 - x[i, j], name=f"alternative_servicing_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_warehouses': 10,
        'max_warehouses': 112,
        'min_locations': 100,
        'max_locations': 600,
        'min_charging_stations': 35,
        'max_charging_stations': 180,
    }

    optimization = WarehouseLocationOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")