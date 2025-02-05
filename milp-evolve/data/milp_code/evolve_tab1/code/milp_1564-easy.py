import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ShipmentDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        weights = np.random.normal(loc=self.weight_mean, scale=self.weight_std, size=self.number_of_deliveries).astype(int)
        profits = weights + np.random.normal(loc=self.profit_mean_shift, scale=self.profit_std, size=self.number_of_deliveries).astype(int)

        weights = np.clip(weights, self.min_range, self.max_range)
        profits = np.clip(profits, self.min_range, self.max_range)

        cargo_volume_capacities = np.zeros(self.number_of_vehicles, dtype=int)
        cargo_volume_capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_vehicles,
                                                         0.6 * weights.sum() // self.number_of_vehicles,
                                                         self.number_of_vehicles - 1)
        cargo_volume_capacities[-1] = 0.5 * weights.sum() - cargo_volume_capacities[:-1].sum()

        handling_times = np.random.uniform(1, 5, self.number_of_vehicles)
        handling_time_penalty = np.random.uniform(20, 50, self.number_of_vehicles)
        delivery_deadlines = np.random.uniform(1, 10, self.number_of_deliveries) 
        vehicle_speeds = np.random.uniform(30, 60, self.number_of_vehicles)
        max_energy_consumption = np.random.uniform(1000, 5000, size=self.number_of_vehicles)
        emission_rates = np.random.uniform(0.1, 1.0, size=(self.number_of_deliveries, self.number_of_vehicles))

        item_weights = np.random.randint(1, 10, size=(self.number_of_deliveries,))
        item_profits = np.random.randint(10, 100, size=(self.number_of_deliveries,))
        
        handling_availability = np.ones((self.number_of_vehicles, self.number_of_periods))  # Assume all vehicles are initially available in all periods
        priority_deliveries = np.random.choice([0, 1], size=self.number_of_deliveries, p=[0.8, 0.2])  # 20% of deliveries are priority

        return {
            'weights': weights, 
            'profits': profits, 
            'cargo_volume_capacities': cargo_volume_capacities,
            'handling_times': handling_times,
            'handling_time_penalty': handling_time_penalty,
            'delivery_deadlines': delivery_deadlines,
            'vehicle_speeds': vehicle_speeds,
            'max_energy_consumption': max_energy_consumption,
            'emission_rates': emission_rates,
            'item_weights': item_weights,
            'item_profits': item_profits,
            'handling_availability': handling_availability,
            'priority_deliveries': priority_deliveries
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        cargo_volume_capacities = instance['cargo_volume_capacities']
        handling_times = instance['handling_times']
        handling_time_penalty = instance['handling_time_penalty']
        delivery_deadlines = instance['delivery_deadlines']
        vehicle_speeds = instance['vehicle_speeds']
        max_energy_consumption = instance['max_energy_consumption']
        emission_rates = instance['emission_rates']
        item_weights = instance['item_weights']
        item_profits = instance['item_profits']
        handling_availability = instance['handling_availability']
        priority_deliveries = instance['priority_deliveries']

        number_of_deliveries = len(weights)
        number_of_vehicles = len(cargo_volume_capacities)
        
        number_of_periods = instance['handling_availability'].shape[1]

        model = Model("ShipmentDeliveryOptimization")
        var_names = {}
        start_times = {}
        z = {}
        energy_vars = {}
        emission_vars = {}
        availability_vars = {}
        priority_vars = {}

        for i in range(number_of_deliveries):
            for j in range(number_of_vehicles):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
                start_times[(i, j)] = model.addVar(vtype="C", lb=0, ub=delivery_deadlines[i], name=f"start_time_{i}_{j}")
                emission_vars[(i, j)] = model.addVar(vtype="C", name=f"emission_{i}_{j}")
            z[i] = model.addVar(vtype="B", name=f"z_{i}")

        for j in range(number_of_vehicles):
            energy_vars[j] = model.addVar(vtype="C", name=f"Energy_{j}")
            for p in range(number_of_periods):
                availability_vars[(j, p)] = model.addVar(vtype="B", name=f"Availability_{j}_{p}")

        for i in range(number_of_deliveries):
            priority_vars[i] = model.addVar(vtype="B", name=f"Priority_{i}")

        objective_expr = quicksum((profits[i] * (j+1)) * var_names[(i, j)] for i in range(number_of_deliveries) for j in range(number_of_vehicles))
        handling_cost = quicksum(handling_time_penalty[j] * handling_times[j] for j in range(number_of_vehicles))
        energy_penalty = quicksum(energy_vars[j] for j in range(number_of_vehicles))
        emission_penalty = quicksum(emission_vars[(i, j)] for i in range(number_of_deliveries) for j in range(number_of_vehicles))
        
        objective_expr -= (handling_cost + energy_penalty + emission_penalty)

        for i in range(number_of_deliveries):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_vehicles)) <= z[i],
                f"DeliveryAssignment_{i}"
            )

        for j in range(number_of_vehicles):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_deliveries)) <= cargo_volume_capacities[j],
                f"VehicleCapacity_{j}"
            )

        for j in range(number_of_vehicles):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_deliveries)) >= 0.1 * cargo_volume_capacities[j],
                f"VehicleMinUtilization_{j}"
            )

        for i in range(number_of_deliveries):
            for j in range(number_of_vehicles):
                model.addCons(
                    start_times[(i, j)] <= delivery_deadlines[i] * var_names[(i, j)],
                    f"TimeWindow_{i}_{j}"
                )
                model.addCons(
                    start_times[(i, j)] >= (delivery_deadlines[i] - (weights[i] / vehicle_speeds[j])) * var_names[(i, j)],
                    f"StartTimeConstraint_{i}_{j}"
                )
        
        for i in range(number_of_deliveries):
            for j in range(number_of_vehicles):
                model.addCons(var_names[(i, j)] <= z[i], f"BigM_constraint_1_{i}_{j}")  
                model.addCons(var_names[(i, j)] >= z[i] - (1 - var_names[(i, j)]), f"BigM_constraint_2_{i}_{j}")  

        for j in range(number_of_vehicles):
            model.addCons(
                energy_vars[j] <= max_energy_consumption[j],
                f"MaxEnergy_{j}"
            )

        for i in range(number_of_deliveries):
            for j in range(number_of_vehicles):
                model.addCons(
                    emission_vars[(i, j)] == emission_rates[(i, j)] * var_names[(i, j)],
                    f"Emission_{i}_{j}"
                )

        for j in range(number_of_vehicles):
            for p in range(number_of_periods):
                model.addCons(
                    availability_vars[(j, p)] == handling_availability[j, p],
                    f"Availability_{j}_{p}"
                )

        for i in range(number_of_deliveries):
            model.addCons(
                priority_vars[i] == priority_deliveries[i],
                f"Priority_{i}"
            )

        for i in range(number_of_deliveries):
            for j in range(number_of_vehicles):
                for p in range(number_of_periods):
                    model.addCons(
                        start_times[(i, j)] <= delivery_deadlines[i] * availability_vars[(j, p)] * var_names[(i, j)],
                        f"TimeWindowPeriods_{i}_{j}_{p}"
                    )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_deliveries': 1500,
        'number_of_vehicles': 4,
        'number_of_periods': 3,
        'min_range': 2,
        'max_range': 336,
        'weight_mean': 60,
        'weight_std': 2500,
        'profit_mean_shift': 750,
        'profit_std': 0,
    }
    
    optimizer = ShipmentDeliveryOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")