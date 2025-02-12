import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum, multidict

class EVChargingStationPlacement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.num_stations > 0 and self.num_regions > 0
        assert self.min_install_cost >= 0 and self.max_install_cost >= self.min_install_cost
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        assert self.grid_limit > 0

        # Generate data for installation costs, charging demands, and station capacities
        install_costs = np.random.randint(self.min_install_cost, self.max_install_cost + 1, self.num_stations)
        charging_demands = np.random.randint(self.min_demand, self.max_demand + 1, self.num_regions)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.num_stations)
        coverage_radius = np.random.randint(self.min_coverage_radius, self.max_coverage_radius + 1, self.num_stations)
        
        regions = np.arange(1, self.num_regions + 1)
        stations = np.arange(1, self.num_stations + 1)

        return {
            "install_costs": install_costs,
            "charging_demands": charging_demands,
            "capacities": capacities,
            "coverage_radius": coverage_radius,
            "regions": regions,
            "stations": stations,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        install_costs = instance['install_costs']
        charging_demands = instance['charging_demands']
        capacities = instance['capacities']
        coverage_radius = instance['coverage_radius']
        regions = instance['regions']
        stations = instance['stations']
        
        model = Model("EVChargingStationPlacement")
        num_stations = len(stations)
        num_regions = len(regions)
        
        # Decision variables
        station_open = {s: model.addVar(vtype="B", name=f"StationOpen_{s}") for s in range(num_stations)}
        region_covered = {(s, r): model.addVar(vtype="B", name=f"Station_{s}_Region_{r}") for s in range(num_stations) for r in range(num_regions)}

        # Objective: minimize the total installation cost
        model.setObjective(
            quicksum(install_costs[s] * station_open[s] for s in range(num_stations)),
            "minimize"
        )

        # Constraints: Total installation cost can't exceed the given budget
        model.addCons(quicksum(install_costs[s] * station_open[s] for s in range(num_stations)) <= self.budget_constraint, "TotalInstallationCost")

        # Constraints: Only open stations can cover regions within their radius
        for s in range(num_stations):
            for r in range(num_regions):
                if np.linalg.norm(np.array([s, 0]) - np.array([r, 0])) > coverage_radius[s]:
                    model.addCons(region_covered[s, r] == 0, f"Region_{r}_OutOfRadius_{s}")
                else:
                    model.addCons(region_covered[s, r] <= station_open[s], f"Station_{s}_Cover_{r}")

        # Constraints: Capacity Limits of Stations
        for s in range(num_stations):
            model.addCons(quicksum(region_covered[s, r] * charging_demands[r] for r in range(num_regions)) <= capacities[s], f"Station_{s}_Capacity")
        
        # Constraints: Total number of covered regions should match demands
        total_demand = np.sum(charging_demands)
        model.addCons(quicksum(region_covered[s, r] * charging_demands[r] for s in range(num_stations) for r in range(num_regions)) >= total_demand, "TotalDemandCoverage")

        # Constraints: Grid capacity limits
        model.addCons(quicksum(capacities[s] * station_open[s] for s in range(num_stations)) <= self.grid_limit, "GridSupplyLimit")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_stations': 100,
        'num_regions': 50,
        'min_install_cost': 5000,
        'max_install_cost': 20000,
        'min_demand': 1200,
        'max_demand': 2400,
        'min_capacity': 2000,
        'max_capacity': 5000,
        'min_coverage_radius': 5,
        'max_coverage_radius': 70,
        'budget_constraint': 500000,
        'grid_limit': 300000,
    }

    station_optimizer = EVChargingStationPlacement(parameters, seed)
    instance = station_optimizer.generate_instance()
    solve_status, solve_time, objective_value = station_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")