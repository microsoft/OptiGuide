import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CapacitatedRenewableDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_data(self):
        city_demands = self.randint(self.n_cities, self.demand_interval)
        plant_capacities = self.randint(self.n_plants, self.capacity_interval)
        setup_costs = self.randint(self.n_plants, self.setup_cost_interval)
        transmission_loss_coeff = self.randint(self.n_cities * self.n_plants, self.loss_coefficient_interval).reshape(self.n_cities, self.n_plants)

        res = {
            'city_demands': city_demands,
            'plant_capacities': plant_capacities,
            'setup_costs': setup_costs,
            'transmission_loss_coeff': transmission_loss_coeff
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        city_demands = instance['city_demands']
        plant_capacities = instance['plant_capacities']
        setup_costs = instance['setup_costs']
        transmission_loss_coeff = instance['transmission_loss_coeff']
        
        n_cities = len(city_demands)
        n_plants = len(plant_capacities)
        
        model = Model("CapacitatedRenewableDeployment")
        
        # Decision variables
        open_plants = {j: model.addVar(vtype="B", name=f"Open_Plant_{j}") for j in range(n_plants)}
        connect = {(i, j): model.addVar(vtype="C", name=f"Connect_{i}_{j}") for i in range(n_cities) for j in range(n_plants)}

        # Objective: minimize the total cost considering setup cost and transmission loss
        objective_expr = quicksum(setup_costs[j] * open_plants[j] for j in range(n_plants)) \
                        + quicksum(transmission_loss_coeff[i, j] * connect[i, j] for i in range(n_cities) for j in range(n_plants))
        
        # Constraints: demand must be met
        for i in range(n_cities):
            model.addCons(quicksum(connect[i, j] for j in range(n_plants)) >= city_demands[i], f"Need_{i}")
        
        # Constraints: capacity limits
        for j in range(n_plants):
            model.addCons(quicksum(connect[i, j] for i in range(n_cities)) <= plant_capacities[j] * open_plants[j], f"Capacity_{j}")
        
        # Ensure correct transmission
        for i in range(n_cities):
            for j in range(n_plants):
                model.addCons(connect[i, j] <= city_demands[i] * open_plants[j], f"EnergyTransmission_{i}_{j}")
                
        total_demand = np.sum(city_demands)
        model.addCons(quicksum(plant_capacities[j] * open_plants[j] for j in range(n_plants)) >= total_demand, "TotalNeed")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_cities': 75,
        'n_plants': 300,
        'demand_interval': (45, 324),
        'capacity_interval': (90, 1449),
        'setup_cost_interval': (700, 1400),
        'loss_coefficient_interval': (3, 30),
    }

    renewable_deployment = CapacitatedRenewableDeployment(parameters, seed=seed)
    instance = renewable_deployment.generate_data()
    solve_status, solve_time = renewable_deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")