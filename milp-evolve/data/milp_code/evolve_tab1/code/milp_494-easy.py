import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ElectricVehicleChargingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_charging_stations(self):
        n_locations = np.random.randint(self.min_stations, self.max_stations)
        stations = np.random.choice(range(2000), size=n_locations, replace=False)  # Random unique charging station IDs
        return stations

    def generate_zones(self, n_stations):
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        zones = np.random.choice(range(3000, 4000), size=n_zones, replace=False)
        demands = {zone: np.random.randint(self.min_demand, self.max_demand) for zone in zones}
        return zones, demands

    def generate_assignment_costs(self, stations, zones):
        costs = {(s, z): np.random.uniform(self.min_assign_cost, self.max_assign_cost) for s in stations for z in zones}
        return costs

    def generate_operational_capacities(self, stations):
        capacities = {s: np.random.randint(self.min_capacity, self.max_capacity) for s in stations}
        return capacities

    def get_instance(self):
        stations = self.generate_charging_stations()
        zones, demands = self.generate_zones(len(stations))
        assign_costs = self.generate_assignment_costs(stations, zones)
        capacities = self.generate_operational_capacities(stations)
        
        install_costs = {s: np.random.uniform(self.min_install_cost, self.max_install_cost) for s in stations}
        operational_costs = {s: np.random.uniform(self.min_operational_cost, self.max_operational_cost) for s in stations}
        extra_costs = {s: np.random.uniform(self.min_extra_cost, self.max_extra_cost) for s in stations}
        
        return {
            'stations': stations,
            'zones': zones,
            'assign_costs': assign_costs,
            'capacities': capacities,
            'demands': demands,
            'install_costs': install_costs,
            'operational_costs': operational_costs,
            'extra_costs': extra_costs,
        }

    def solve(self, instance):
        stations = instance['stations']
        zones = instance['zones']
        assign_costs = instance['assign_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        install_costs = instance['install_costs']
        operational_costs = instance['operational_costs']
        extra_costs = instance['extra_costs']

        model = Model("ElectricVehicleChargingOptimization")

        FacilityLocation_vars = {s: model.addVar(vtype="B", name=f"Facility_Loc_{s}") for s in stations}
        NetworkAllocation_vars = {(s, z): model.addVar(vtype="B", name=f"NetworkAlloc_{s}_{z}") for s in stations for z in zones}
        ChargingUsage_vars = {(s, z): model.addVar(vtype="C", name=f"ChargingUsage_{s}_{z}") for s in stations for z in zones}
        TotalNetworkCapacity_vars = {s: model.addVar(vtype="C", name=f"TotalNetworkCapacity_{s}") for s in stations}
        TemporaryArrangement_vars = {s: model.addVar(vtype="C", name=f"TempArrangement_{s}") for s in stations}

        # Objective function
        total_cost = quicksum(FacilityLocation_vars[s] * install_costs[s] for s in stations)
        total_cost += quicksum(NetworkAllocation_vars[s, z] * assign_costs[s, z] for s in stations for z in zones)
        total_cost += quicksum(TotalNetworkCapacity_vars[s] * operational_costs[s] for s in stations)
        total_cost += quicksum(TemporaryArrangement_vars[s] * extra_costs[s] for s in stations)
        
        model.setObjective(total_cost, "minimize")

        # Constraints
        for s in stations:
            model.addCons(
                quicksum(ChargingUsage_vars[s, z] for z in zones) <= capacities[s] * FacilityLocation_vars[s],
                name=f"Capacity_{s}"
            )

        for z in zones:
            model.addCons(
                quicksum(ChargingUsage_vars[s, z] for s in stations) >= demands[z],
                name=f"ZoneDemand_{z}"
            )

        for s in stations:
            model.addCons(
                TotalNetworkCapacity_vars[s] <= capacities[s],
                name=f"NetworkCapacityLimit_{s}"
            )

        for s in stations:
            model.addCons(
                TemporaryArrangement_vars[s] <= capacities[s] * 0.2,  # Assume temporary arrangement is limited to 20% of the capacity
                name=f"TempArrangementLimit_{s}"
            )

        model.setParam('limits/time', 10 * 60)  # Set a time limit of 10 minutes for solving
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_stations': 25,
        'max_stations': 600,
        'min_zones': 210,
        'max_zones': 300,
        'min_demand': 700,
        'max_demand': 800,
        'min_assign_cost': 7.0,
        'max_assign_cost': 500.0,
        'min_capacity': 750,
        'max_capacity': 2000,
        'min_install_cost': 25000,
        'max_install_cost': 200000,
        'min_operational_cost': 3000,
        'max_operational_cost': 5000,
        'min_extra_cost': 400,
        'max_extra_cost': 3000,
    }

    optimizer = ElectricVehicleChargingOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")