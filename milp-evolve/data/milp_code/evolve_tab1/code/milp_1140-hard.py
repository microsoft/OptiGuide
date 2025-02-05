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
        assert self.NumberOfDrones > 0 and self.PackagesPerMunicipality > 0
        assert self.FuelCostRange[0] >= 0 and self.FuelCostRange[1] >= self.FuelCostRange[0]
        assert self.DroneCapacityRange[0] > 0 and self.DroneCapacityRange[1] >= self.DroneCapacityRange[0]

        operational_costs = np.random.randint(self.FuelCostRange[0], self.FuelCostRange[1] + 1, self.NumberOfDrones)
        transit_costs = np.random.randint(self.FuelCostRange[0], self.FuelCostRange[1] + 1, (self.NumberOfDrones, self.PackagesPerMunicipality))
        capacities = np.random.randint(self.DroneCapacityRange[0], self.DroneCapacityRange[1] + 1, self.NumberOfDrones)
        package_demands = np.random.randint(self.PackageDemandRange[0], self.PackageDemandRange[1] + 1, self.PackagesPerMunicipality)
        
        return {
            "operational_costs": operational_costs,
            "transit_costs": transit_costs,
            "capacities": capacities,
            "package_demands": package_demands
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        transit_costs = instance['transit_costs']
        capacities = instance['capacities']
        package_demands = instance['package_demands']

        model = Model("DroneDeliveryOptimization")
        n_drones = len(operational_costs)
        n_packages = len(package_demands)

        drone_vars = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(n_drones)}
        package_assignment_vars = {(d, p): model.addVar(vtype="C", name=f"Package_{d}_Package_{p}") for d in range(n_drones) for p in range(n_packages)}

        model.setObjective(
            quicksum(operational_costs[d] * drone_vars[d] for d in range(n_drones)) +
            quicksum(transit_costs[d][p] * package_assignment_vars[d, p] for d in range(n_drones) for p in range(n_packages)),
            "minimize"
        )

        # Constraints
        # Package demand satisfaction (total deliveries must cover total demand)
        for p in range(n_packages):
            model.addCons(quicksum(package_assignment_vars[d, p] for d in range(n_drones)) == package_demands[p], f"Package_Demand_Satisfaction_{p}")

        # Capacity limits for each drone
        for d in range(n_drones):
            model.addCons(quicksum(package_assignment_vars[d, p] for p in range(n_packages)) <= capacities[d] * drone_vars[d], f"Drone_Capacity_{d}")

        # Package assignment only if drone is operational
        for d in range(n_drones):
            for p in range(n_packages):
                model.addCons(package_assignment_vars[d, p] <= package_demands[p] * drone_vars[d], f"Operational_Constraint_{d}_{p}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'NumberOfDrones': 500,
        'PackagesPerMunicipality': 75,
        'FuelCostRange': (75, 110),
        'DroneCapacityRange': (250, 375),
        'PackageDemandRange': (400, 2000),
    }
    
    drone_optimizer = DroneDeliveryOptimization(parameters, seed=seed)
    instance = drone_optimizer.generate_instance()
    solve_status, solve_time, objective_value = drone_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")