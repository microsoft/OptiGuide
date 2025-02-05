import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class RenewableEnergyAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_energy_plants = random.randint(self.min_energy_plants, self.max_energy_plants)
        num_cities = random.randint(self.min_cities, self.max_cities)

        transmission_costs = np.random.randint(50, 300, size=(num_cities, num_energy_plants))
        fixed_costs = np.random.randint(1000, 5000, size=num_energy_plants)
        city_demand = np.random.randint(100, 500, size=num_cities)
        energy_generation_capacity = np.random.randint(1000, 5000, size=num_energy_plants)

        res = {
            'num_energy_plants': num_energy_plants,
            'num_cities': num_cities,
            'transmission_costs': transmission_costs,
            'fixed_costs': fixed_costs,
            'city_demand': city_demand,
            'energy_generation_capacity': energy_generation_capacity,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_energy_plants = instance['num_energy_plants']
        num_cities = instance['num_cities']
        transmission_costs = instance['transmission_costs']
        fixed_costs = instance['fixed_costs']
        city_demand = instance['city_demand']
        energy_generation_capacity = instance['energy_generation_capacity']

        model = Model("RenewableEnergyAllocation")

        # Variables
        energy_plant = {j: model.addVar(vtype="B", name=f"energy_plant_{j}") for j in range(num_energy_plants)}
        transmission = {(i, j): model.addVar(vtype="B", name=f"transmission_{i}_{j}") for i in range(num_cities) for j in range(num_energy_plants)}

        # Objective function: Minimize total costs
        total_cost = quicksum(transmission[i, j] * transmission_costs[i, j] for i in range(num_cities) for j in range(num_energy_plants)) + \
                     quicksum(energy_plant[j] * fixed_costs[j] for j in range(num_energy_plants))
        
        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_cities):
            model.addCons(quicksum(transmission[i, j] for j in range(num_energy_plants)) >= 1, name=f"MinimumEnergySupply_{i}")

        for j in range(num_energy_plants):
            for i in range(num_cities):
                model.addCons(transmission[i, j] <= energy_plant[j], name=f"CityConnection_{i}_{j}")

        for j in range(num_energy_plants):
            model.addCons(quicksum(transmission[i, j] * city_demand[i] for i in range(num_cities)) <= energy_generation_capacity[j], name=f"HubCapacity_{j}")

        # New Symmetry Breaking Constraints
        for j in range(num_energy_plants - 1):
            model.addCons(energy_plant[j] >= energy_plant[j + 1], name=f"SymmetryBreaking_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_energy_plants': 25,
        'max_energy_plants': 600,
        'min_cities': 24,
        'max_cities': 600,
    }
    
    optimization = RenewableEnergyAllocation(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")