import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EnergyNetwork:
    def __init__(self, number_of_zones, supply_capacities, emission_limits):
        self.number_of_zones = number_of_zones
        self.zones = np.arange(number_of_zones)
        self.supply_capacities = supply_capacities
        self.emission_limits = emission_limits

class EnergyAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        installation_costs = np.random.randint(self.min_installation_cost, self.max_installation_cost + 1, self.n_zones)
        energy_costs = np.random.normal(self.mean_energy_cost, self.std_dev_energy_cost, (self.n_zones, self.n_ev))
        supply_capacities = np.random.randint(self.min_zone_capacity, self.max_zone_capacity + 1, self.n_zones)
        ev_energy_requirements = np.random.gamma(2, 5, self.n_ev).astype(int)
        emission_limits = np.random.uniform(self.min_emission_limit, self.max_emission_limit, self.n_zones)

        return {
            "installation_costs": installation_costs,
            "energy_costs": energy_costs,
            "supply_capacities": supply_capacities,
            "ev_energy_requirements": ev_energy_requirements,
            "emission_limits": emission_limits,
        }

    def solve(self, instance):
        installation_costs = instance['installation_costs']
        energy_costs = instance['energy_costs']
        supply_capacities = instance['supply_capacities']
        ev_energy_requirements = instance['ev_energy_requirements']
        emission_limits = instance['emission_limits']

        model = Model("EnergyAllocation")
        n_zones = len(installation_costs)
        n_ev = len(ev_energy_requirements)

        charging_station_vars = {z: model.addVar(vtype="B", name=f"ChargingStation_{z}") for z in range(n_zones)}
        energy_allocation_vars = {(z, e): model.addVar(vtype="C", name=f"Zone_{z}_EV_{e}") for z in range(n_zones) for e in range(n_ev)}

        # Objective function
        model.setObjective(
            quicksum(installation_costs[z] * charging_station_vars[z] for z in range(n_zones)) +
            quicksum(energy_costs[z, e] * energy_allocation_vars[z, e] for z in range(n_zones) for e in range(n_ev)),
            "minimize"
        )

        # Constraints
        for e in range(n_ev):
            model.addCons(quicksum(energy_allocation_vars[z, e] for z in range(n_zones)) >= ev_energy_requirements[e], f"Vehicle_{e}_MinimumEnergy_Requirement")

        for z in range(n_zones):
            model.addCons(quicksum(energy_allocation_vars[z, e] for e in range(n_ev)) <= supply_capacities[z], f"Zone_{z}_EnergyCapacity")

        for z in range(n_zones):
            model.addCons(
                quicksum(energy_allocation_vars[z, e] for e in range(n_ev)) <= emission_limits[z] * charging_station_vars[z],
                f"Emission_Limit_{z}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_zones': 50,
        'n_ev': 225,
        'min_installation_cost': 2000,
        'max_installation_cost': 5000,
        'mean_energy_cost': 1000,
        'std_dev_energy_cost': 20,
        'min_zone_capacity': 300,
        'max_zone_capacity': 2000,
        'min_emission_limit': 100,
        'max_emission_limit': 1800,
    }

    energy_optimizer = EnergyAllocation(parameters, seed=seed)
    instance = energy_optimizer.generate_instance()
    solve_status, solve_time, objective_value = energy_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")