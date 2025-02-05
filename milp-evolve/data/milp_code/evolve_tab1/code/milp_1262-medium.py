import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NeighboorhoodElectricVehicleOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_locations > 0 and self.n_neighborhoods > 0
        assert self.min_construction_cost >= 0 and self.max_construction_cost >= self.min_construction_cost
        assert self.min_energy_cost >= 0 and self.max_energy_cost >= self.min_energy_cost
        assert self.min_demand > 0 and self.max_demand >= self.min_demand
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        assert self.max_weight > 0

        construction_costs = np.random.randint(self.min_construction_cost, self.max_construction_cost + 1, self.n_locations)
        energy_costs = np.random.randint(self.min_energy_cost, self.max_energy_cost + 1, (self.n_locations, self.n_neighborhoods))
        neighborhood_demands = np.random.randint(self.min_demand, self.max_demand + 1, self.n_neighborhoods)
        location_capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_locations)
        penalty_costs = np.random.uniform(50, 150, self.n_neighborhoods)
        max_weights = np.random.randint(1, self.max_weight + 1, self.n_locations)  # in kg
        
        return {
            "construction_costs": construction_costs,
            "energy_costs": energy_costs,
            "neighborhood_demands": neighborhood_demands,
            "location_capacities": location_capacities,
            "penalty_costs": penalty_costs,
            "max_weights": max_weights,
        }

    def solve(self, instance):
        construction_costs = instance['construction_costs']
        energy_costs = instance['energy_costs']
        neighborhood_demands = instance['neighborhood_demands']
        location_capacities = instance['location_capacities']
        penalty_costs = instance['penalty_costs']
        max_weights = instance['max_weights']

        model = Model("NeighboorhoodElectricVehicleOptimization")
        n_locations = len(construction_costs)
        n_neighborhoods = len(neighborhood_demands)

        station_vars = {(l): model.addVar(vtype="B", name=f"Station_Location_{l}") for l in range(n_locations)}
        energy_vars = {(l, n): model.addVar(vtype="I", name=f"Energy_Location_{l}_Neighborhood_{n}") for l in range(n_locations) for n in range(n_neighborhoods)}
        unmet_demand_vars = {(n): model.addVar(vtype="I", name=f"Unmet_Neighborhood_{n}") for n in range(n_neighborhoods)}

        # Objective Function
        model.setObjective(
            quicksum(construction_costs[l] * station_vars[l] for l in range(n_locations)) +
            quicksum(energy_costs[l][n] * energy_vars[l, n] for l in range(n_locations) for n in range(n_neighborhoods)) +
            quicksum(penalty_costs[n] * unmet_demand_vars[n] for n in range(n_neighborhoods)),
            "minimize"
        )

        # Constraints
        for n in range(n_neighborhoods):
            model.addCons(
                quicksum(energy_vars[l, n] for l in range(n_locations)) + unmet_demand_vars[n] == neighborhood_demands[n],
                f"Demand_Fulfillment_Neighborhood_{n}"
            )

        for l in range(n_locations):
            model.addCons(
                quicksum(energy_vars[l, n] for n in range(n_neighborhoods)) <= location_capacities[l] * station_vars[l],
                f"Station_Capacity_{l}"
            )
            
            model.addCons(
                (station_vars[l] * max_weights[l]) <= self.max_weight,
                f"Weight_Limit_Station_{l}"
            )

        model.addCons(
            quicksum(energy_vars[l, n] for l in range(n_locations) for n in range(n_neighborhoods)) <= self.total_grid_capacity,
            "Grid_Capacity"
        )
        
        # Set Covering Constraint
        for n in range(n_neighborhoods):
            model.addCons(
                quicksum(station_vars[l] for l in range(n_locations)) >= 1,
                f"Set_Covering_Neighborhood_{n}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 200,
        'n_neighborhoods': 160,
        'min_construction_cost': 3000,
        'max_construction_cost': 5000,
        'min_energy_cost': 120,
        'max_energy_cost': 300,
        'min_demand': 800,
        'max_demand': 3000,
        'min_capacity': 2000,
        'max_capacity': 2000,
        'max_weight': 10000,
        'total_grid_capacity': 50000,
    }

    ev_optimizer = NeighboorhoodElectricVehicleOptimization(parameters, seed=42)
    instance = ev_optimizer.generate_instance()
    solve_status, solve_time, objective_value = ev_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")