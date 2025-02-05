import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EnergyGridOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_plants = random.randint(self.min_plants, self.max_plants)
        n_regions = random.randint(self.min_regions, self.max_regions)
        
        # Costs and capacities
        plant_setup_cost = np.random.randint(200, 1000, size=n_plants)
        production_cost = np.random.randint(50, 200, size=n_plants)
        transport_cost = np.random.randint(5, 50, size=(n_regions, n_plants))
        plant_capacity = np.random.randint(100, 500, size=n_plants)
        monthly_demand = np.random.randint(30, 150, size=n_regions)
        
        # Additional data
        hot_spot_areas = np.random.randint(0, 2, size=n_regions)

        res = {
            'n_plants': n_plants,
            'n_regions': n_regions,
            'plant_setup_cost': plant_setup_cost,
            'production_cost': production_cost,
            'transport_cost': transport_cost,
            'plant_capacity': plant_capacity,
            'monthly_demand': monthly_demand,
            'hot_spot_areas': hot_spot_areas,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_plants = instance['n_plants']
        n_regions = instance['n_regions']
        plant_setup_cost = instance['plant_setup_cost']
        production_cost = instance['production_cost']
        transport_cost = instance['transport_cost']
        plant_capacity = instance['plant_capacity']
        monthly_demand = instance['monthly_demand']
        hot_spot_areas = instance['hot_spot_areas']
        
        model = Model("EnergyGridOptimization")
        
        # Variables
        plant_open = {j: model.addVar(vtype="B", name=f"plant_open_{j}") for j in range(n_plants)}
        region_assignment = {}
        for i in range(n_regions):
            for j in range(n_plants):
                region_assignment[i, j] = model.addVar(vtype="B", name=f"region_assignment_{i}_{j}")

        # Objective function: Minimize total setup, production and transportation costs
        total_cost = quicksum(plant_open[j] * plant_setup_cost[j] for j in range(n_plants)) + \
                     quicksum(region_assignment[i, j] * transport_cost[i, j] * monthly_demand[i] for i in range(n_regions) for j in range(n_plants)) + \
                     quicksum(plant_open[j] * production_cost[j] for j in range(n_plants))
        model.setObjective(total_cost, "minimize")

        # Constraints
        # Each region must receive energy from exactly one plant
        for i in range(n_regions):
            model.addCons(quicksum(region_assignment[i, j] for j in range(n_plants)) == 1, name=f"region_assignment_{i}")
        
        # Plant capacity constraints
        for j in range(n_plants):
            model.addCons(quicksum(region_assignment[i, j] * monthly_demand[i] for i in range(n_regions)) <= plant_capacity[j] * plant_open[j],
                          name=f"plant_capacity_{j}")
        
        # Regions can only receive energy from open plants
        for i in range(n_regions):
            for j in range(n_plants):
                model.addCons(region_assignment[i, j] <= plant_open[j], name=f"only_open_plants_{i}_{j}")
        
        # Ensure each hot spot area gets energy from at least one plant
        for i in range(n_regions):
            if hot_spot_areas[i]:
                model.addCons(quicksum(region_assignment[i, j] for j in range(n_plants)) >= 1, name=f"hot_spot_area_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 456
    parameters = {
        'min_plants': 10,
        'max_plants': 80,
        'min_regions': 20,
        'max_regions': 150,
    }
    
    energy_grid_optimizer = EnergyGridOptimization(parameters, seed=seed)
    instance = energy_grid_optimizer.generate_instance()
    solve_status, solve_time = energy_grid_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")