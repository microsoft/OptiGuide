import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DroneDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_feeder_stations > 0 and self.n_drones > 0
        assert self.min_energy_cost >= 0 and self.max_energy_cost >= self.min_energy_cost
        assert self.min_handling_cost >= 0 and self.max_handling_cost >= self.min_handling_cost

        energy_costs = np.random.randint(self.min_energy_cost, self.max_energy_cost + 1, (self.n_drones, self.n_feeder_stations))
        handling_costs = np.random.randint(self.min_handling_cost, self.max_handling_cost + 1, self.n_feeder_stations)
        feeder_demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_feeder_stations).astype(int)
        drone_capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_drones)

        return {
            "energy_costs": energy_costs,
            "handling_costs": handling_costs,
            "feeder_demands": feeder_demands,
            "drone_capacities": drone_capacities
        }

    def solve(self, instance):
        energy_costs = instance["energy_costs"]
        handling_costs = instance["handling_costs"]
        feeder_demands = instance["feeder_demands"]
        drone_capacities = instance["drone_capacities"]

        model = Model("DroneDeliveryOptimization")
        n_drones = len(drone_capacities)
        n_feeder_stations = len(feeder_demands)

        # Decision variables
        number_drones = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(n_drones)}
        assignments = {(d, f): model.addVar(vtype="B", name=f"Drone_{d}_Feeder_{f}") for d in range(n_drones) for f in range(n_feeder_stations)}

        # Objective function
        model.setObjective(
            quicksum(energy_costs[d, f] * assignments[d, f] for d in range(n_drones) for f in range(n_feeder_stations)) +
            quicksum(handling_costs[f] * quicksum(assignments[d, f] for d in range(n_drones)) for f in range(n_feeder_stations)),
            "minimize"
        )

        # Constraints: Each feeder station must be serviced by at least one drone
        for f in range(n_feeder_stations):
            model.addCons(quicksum(assignments[d, f] for d in range(n_drones)) >= 1, f"Feeder_{f}_Coverage")

        # Constraints: Total energy consumption by each drone cannot exceed its capacity
        for d in range(n_drones):
            model.addCons(quicksum(feeder_demands[f] * assignments[d, f] for f in range(n_feeder_stations)) <= drone_capacities[d], f"Drone_{d}_Capacity")

        # Symmetry breaking constraints: Enforce ordered use of drones
        for d in range(1, n_drones):
            model.addCons(number_drones[d] <= number_drones[d - 1], f"Drone_{d}_Symmetry_Breaking")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value


if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_drones': 80,
        'n_feeder_stations': 150,
        'min_energy_cost': 300,
        'max_energy_cost': 2000,
        'min_handling_cost': 350,
        'max_handling_cost': 1800,
        'mean_demand': 1200,
        'std_dev_demand': 1050,
        'min_capacity': 1000,
        'max_capacity': 5000,
    }

    drone_optimizer = DroneDeliveryOptimization(parameters, seed)
    instance = drone_optimizer.generate_instance()
    solve_status, solve_time, objective_value = drone_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")