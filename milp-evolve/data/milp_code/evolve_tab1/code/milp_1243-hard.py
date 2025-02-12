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
        weather_effects = np.random.uniform(0.8, 1.2, self.NumberOfDrones)
        drone_reliability = np.random.uniform(0.8, 1.0, self.NumberOfDrones)

        return {
            "operational_costs": operational_costs,
            "transit_costs": transit_costs,
            "transit_times": transit_times,
            "capacities": capacities,
            "package_demands": package_demands,
            "time_windows": time_windows,
            "drone_speeds": drone_speeds,
            "battery_usage_rates": battery_usage_rates,
            "weather_effects": weather_effects,
            "drone_reliability": drone_reliability,
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
        weather_effects = instance['weather_effects']
        drone_reliability = instance['drone_reliability']

        model = Model("DroneDeliveryOptimization")
        n_drones = len(operational_costs)
        n_packages = len(package_demands)

        drone_vars = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(n_drones)}
        package_assignment_vars = {(d, p): model.addVar(vtype="B", name=f"Package_{d}_Package_{p}") for d in range(n_drones) for p in range(n_packages)}
        max_delivery_time = model.addVar(vtype="C", name="Max_Delivery_Time")
        penalty_vars = {(d, p): model.addVar(vtype="C", name=f"Penalty_{d}_{p}") for d in range(n_drones) for p in range(n_packages)}

        model.setObjective(
            quicksum(operational_costs[d] * drone_vars[d] for d in range(n_drones)) +
            quicksum(transit_costs[d][p] * package_assignment_vars[d, p] for d in range(n_drones) for p in range(n_packages)) +
            quicksum(penalty_vars[d, p] for d in range(n_drones) for p in range(n_packages)) +
            max_delivery_time * 10 - quicksum(drone_reliability[d] for d in range(n_drones)),
            "minimize"
        )

        # Constraints
        # Package demand satisfaction (total deliveries must cover total demand)
        for p in range(n_packages):
            model.addCons(quicksum(package_assignment_vars[d, p] for d in range(n_drones)) == package_demands[p], f"Package_Demand_Satisfaction_{p}")

        # Capacity limits for each drone
        for d in range(n_drones):
            model.addCons(quicksum(package_assignment_vars[d, p] for p in range(n_packages)) <= capacities[d] * drone_vars[d], f"Drone_Capacity_{d}")

        # Time window constraints with penalties for late deliveries
        for p in range(n_packages):
            for d in range(n_drones):
                model.addCons(package_assignment_vars[d, p] * transit_times[d][p] * weather_effects[d] <= time_windows[p] + penalty_vars[d, p], f"Time_Window_{d}_{p}")

        # Max delivery time constraint
        for d in range(n_drones):
            for p in range(n_packages):
                model.addCons(package_assignment_vars[d, p] * transit_times[d][p] * weather_effects[d] <= max_delivery_time, f"Max_Delivery_Time_Constraint_{d}_{p}")

        # Ensuring minimal usage of drones to avoid monopolistic delivery
        for d in range(n_drones):
            model.addCons(drone_vars[d] >= quicksum(package_assignment_vars[d, p] for p in range(n_packages)) / capacities[d], f"Minimal_Drone_Usage_{d}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'NumberOfDrones': 750,
        'PackagesPerMunicipality': 13,
        'FuelCostRange': (270, 540),
        'DroneCapacityRange': (843, 1406),
        'PackageDemandRange': (1, 7),
    }

    drone_optimizer = DroneDeliveryOptimization(parameters, seed=seed)
    instance = drone_optimizer.get_instance()
    solve_status, solve_time, objective_value = drone_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")