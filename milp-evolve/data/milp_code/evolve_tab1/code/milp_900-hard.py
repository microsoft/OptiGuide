import random
import time
import numpy as np
from pyscipopt import Model, quicksum, multidict

class EVChargingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        # Parameters based on input specification
        num_stations = random.randint(self.min_stations, self.max_stations)
        num_EV_types = random.randint(self.min_EV_types, self.max_EV_types)
        
        # Cost matrices
        construction_cost = np.random.randint(100000, 500000, size=num_stations)
        maintenance_cost = np.random.randint(5000, 20000, size=num_stations)
        energy_price = np.random.uniform(0.1, 0.5, size=(num_EV_types, num_stations))
        
        # EV demand and usage
        EV_demand = np.random.randint(100, 1000, size=num_EV_types)
        EV_usage_variability = np.random.uniform(0.1, 0.5, size=num_EV_types)
        
        # Station capacity
        station_capacity = np.random.randint(1000, 5000, size=num_stations)
        
        # Budget constraints
        total_budget = np.random.randint(1000000, 5000000)
        
        # Transportation Costs
        transportation_cost = np.random.randint(100, 500, size=(num_EV_types, num_stations))
        
        # V2G related costs and limits
        V2G_cost = np.random.randint(50, 200, size=num_stations)
        V2G_limit = np.random.randint(500, 2000, size=num_stations)
        
        res = {
            'num_stations': num_stations,
            'num_EV_types': num_EV_types,
            'construction_cost': construction_cost,
            'maintenance_cost': maintenance_cost,
            'energy_price': energy_price,
            'EV_demand': EV_demand,
            'EV_usage_variability': EV_usage_variability,
            'station_capacity': station_capacity,
            'total_budget': total_budget,
            'transportation_cost': transportation_cost,
            'V2G_cost': V2G_cost,
            'V2G_limit': V2G_limit,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_stations = instance['num_stations']
        num_EV_types = instance['num_EV_types']
        construction_cost = instance['construction_cost']
        maintenance_cost = instance['maintenance_cost']
        energy_price = instance['energy_price']
        EV_demand = instance['EV_demand']
        EV_usage_variability = instance['EV_usage_variability']
        station_capacity = instance['station_capacity']
        total_budget = instance['total_budget']
        transportation_cost = instance['transportation_cost']
        V2G_cost = instance['V2G_cost']
        V2G_limit = instance['V2G_limit']

        M = 1e6  # Big M constant

        model = Model("EVChargingOptimization")

        # Variables
        FacilityConstruction = {j: model.addVar(vtype="B", name=f"FacilityConstruction_{j}") for j in range(num_stations)}
        EnergyTransmission = {(i, j): model.addVar(vtype="I", name=f"EnergyTransmission_{i}_{j}") for i in range(num_EV_types) for j in range(num_stations)}
        V2GOperation = {j: model.addVar(vtype="B", name=f"V2GOperation_{j}") for j in range(num_stations)}

        # Objective function: Minimize total costs, including construction, maintenance, energy, and transportation costs
        TotalCost = quicksum(FacilityConstruction[j] * construction_cost[j] for j in range(num_stations)) + \
                    quicksum(EnergyTransmission[i, j] * energy_price[i, j] for i in range(num_EV_types) for j in range(num_stations)) + \
                    quicksum(FacilityConstruction[j] * maintenance_cost[j] for j in range(num_stations)) + \
                    quicksum(EnergyTransmission[i, j] * transportation_cost[i, j] for i in range(num_EV_types) for j in range(num_stations)) + \
                    quicksum(V2GOperation[j] * V2G_cost[j] for j in range(num_stations))

        model.setObjective(TotalCost, "minimize")

        # EV demand constraints (considering variability)
        for i in range(num_EV_types):
            demand_min = EV_demand[i] * (1 - EV_usage_variability[i])
            demand_max = EV_demand[i] * (1 + EV_usage_variability[i])
            model.addCons(quicksum(EnergyTransmission[i, j] for j in range(num_stations)) >= demand_min, name=f"EV_demand_min_{i}")
            model.addCons(quicksum(EnergyTransmission[i, j] for j in range(num_stations)) <= demand_max, name=f"EV_demand_max_{i}")

        # Station capacity constraints
        for j in range(num_stations):
            model.addCons(quicksum(EnergyTransmission[i, j] for i in range(num_EV_types)) <= station_capacity[j], name=f"station_capacity_{j}")

        # Facility activity constraint
        for j in range(num_stations):
            activity_coef = sum(EV_demand)
            model.addCons(FacilityConstruction[j] * activity_coef >= quicksum(EnergyTransmission[i, j] for i in range(num_EV_types)), name=f"facility_activity_{j}")

        # Budget constraints
        model.addCons(quicksum(FacilityConstruction[j] * construction_cost[j] for j in range(num_stations)) + \
                      quicksum(EnergyTransmission[i, j] * energy_price[i, j] for i in range(num_EV_types) for j in range(num_stations)) <= total_budget, name="budget_constraint")

        # V2G limit constraints
        for j in range(num_stations):
            model.addCons(quicksum(EnergyTransmission[i, j] for i in range(num_EV_types)) <= M * (1 - V2GOperation[j]) + V2G_limit[j], name=f"V2G_limit_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_stations': 45,
        'max_stations': 700,
        'min_EV_types': 6,
        'max_EV_types': 50,
        'demand_uncertainty_factor': 0.8,
    }

    optimization = EVChargingOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")