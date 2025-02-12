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

    def get_instance(self):
        assert self.NumberOfDrones > 0 and self.PackagesPerMunicipality > 0
        assert self.FuelCostRange[0] >= 0 and self.FuelCostRange[1] >= self.FuelCostRange[0]
        assert self.DroneCapacityRange[0] > 0 and self.DroneCapacityRange[1] >= self.DroneCapacityRange[0]

        operational_costs = np.random.randint(self.FuelCostRange[0], self.FuelCostRange[1] + 1, self.NumberOfDrones)
        transit_costs = np.random.normal(loc=50, scale=10, size=(self.NumberOfDrones, self.PackagesPerMunicipality))
        transit_times = np.random.normal(loc=30, scale=5, size=(self.NumberOfDrones, self.PackagesPerMunicipality))
        capacities = np.random.randint(self.DroneCapacityRange[0], self.DroneCapacityRange[1] + 1, self.NumberOfDrones)
        package_demands = np.random.randint(self.PackageDemandRange[0], self.PackageDemandRange[1] + 1, self.PackagesPerMunicipality)
        time_windows = np.random.randint(30, 100, size=self.PackagesPerMunicipality)
        drone_speeds = np.random.uniform(30, 60, self.NumberOfDrones)
        battery_usage_rates = np.random.uniform(0.5, 1.5, self.NumberOfDrones)

        return {
            "operational_costs": operational_costs,
            "transit_costs": transit_costs,
            "transit_times": transit_times,
            "capacities": capacities,
            "package_demands": package_demands,
            "time_windows": time_windows,
            "drone_speeds": drone_speeds,
            "battery_usage_rates": battery_usage_rates
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        transit_costs = instance['transit_costs']
        transit_times = instance['transit_times']
        capacities = instance['capacities']
        package_demands = instance['package_demands']
        time_windows = instance['time_windows']
        drone_speeds = instance['drone_speeds']
        battery_usage_rates = instance['battery_usage_rates']

        model = Model("DroneDeliveryOptimization")
        n_drones = len(operational_costs)
        n_packages = len(package_demands)

        drone_vars = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(n_drones)}
        package_assignment_vars = {(d, p): model.addVar(vtype="B", name=f"Package_{d}_Package_{p}") for d in range(n_drones) for p in range(n_packages)}
        battery_vars = {d: model.addVar(vtype="C", name=f"Battery_{d}") for d in range(n_drones)}
        max_delivery_time = model.addVar(vtype="C", name="Max_Delivery_Time")

        model.setObjective(
            quicksum(operational_costs[d] * drone_vars[d] for d in range(n_drones)) +
            quicksum(transit_costs[d][p] * package_assignment_vars[d, p] for d in range(n_drones) for p in range(n_packages)) +
            max_delivery_time,
            "minimize"
        )

        # Constraints
        # Package demand satisfaction (total deliveries must cover total demand)
        for p in range(n_packages):
            model.addCons(quicksum(package_assignment_vars[d, p] for d in range(n_drones)) == package_demands[p], f"Package_Demand_Satisfaction_{p}")

        # Capacity limits for each drone
        for d in range(n_drones):
            model.addCons(quicksum(package_assignment_vars[d, p] for p in range(n_packages)) <= capacities[d] * drone_vars[d], f"Drone_Capacity_{d}")

        # Battery constraints for each drone
        for d in range(n_drones):
            model.addCons(battery_vars[d] == quicksum(battery_usage_rates[d] * package_assignment_vars[d, p] * transit_times[d][p] for p in range(n_packages)), f"Drone_Battery_{d}")

        # Time window constraints for deliveries
        for p in range(n_packages):
            for d in range(n_drones):
                model.addCons(package_assignment_vars[d, p] * transit_times[d][p] <= time_windows[p], f"Time_Window_{d}_{p}")

        # Max delivery time constraint
        for d in range(n_drones):
            for p in range(n_packages):
                model.addCons(package_assignment_vars[d, p] * transit_times[d][p] <= max_delivery_time, f"Max_Delivery_Time_Constraint_{d}_{p}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'NumberOfDrones': 1000,
        'PackagesPerMunicipality': 18,
        'FuelCostRange': (270, 540),
        'DroneCapacityRange': (1125, 1875),
        'PackageDemandRange': (15, 75),
    }

    drone_optimizer = DroneDeliveryOptimization(parameters, seed=seed)
    instance = drone_optimizer.get_instance()
    solve_status, solve_time, objective_value = drone_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")