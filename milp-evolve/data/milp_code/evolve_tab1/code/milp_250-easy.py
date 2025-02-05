import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EVChargingStationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        return np.random.rand(self.n_customers, self.n_stations) * self.transport_cost_scale

    def renewable_energy_supply(self):
        return np.random.rand(self.n_renewables) * self.renewable_capacity_scale

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        station_capacities = self.randint(self.n_stations, self.station_capacity_interval)
        renewable_capacities = self.renewable_energy_supply()
        fixed_costs = self.randint(self.n_stations, self.fixed_cost_interval)
        transport_costs = self.unit_transportation_costs()
        
        # Generate essential items
        essential_items = random.sample(range(self.number_of_items), self.number_of_essential_items)

        res = {
            'demands': demands,
            'station_capacities': station_capacities,
            'renewable_capacities': renewable_capacities,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs,
            'essential_items': essential_items,
            'item_profits': self.randint(self.number_of_items, (10, 100)),
            'item_weights': self.randint(self.number_of_items, (1, 10))
        }
        
        return res

    def solve(self, instance):
        demands = instance['demands']
        station_capacities = instance['station_capacities']
        renewable_capacities = instance['renewable_capacities']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        essential_items = instance['essential_items']
        item_profits = instance['item_profits']
        item_weights = instance['item_weights']

        n_customers, n_stations, n_renewables = len(demands), len(station_capacities), len(renewable_capacities)
        number_of_items = len(item_profits)

        model = Model("EVChargingStationOptimization")

        open_stations = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_stations)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(n_customers) for j in range(n_stations)}
        renewable_supply = {j: model.addVar(vtype="C", name=f"RenewableSupply_{j}") for j in range(n_renewables)}
        knapsack_vars = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(number_of_items) for j in range(n_stations)}

        objective_expr = quicksum(fixed_costs[j] * open_stations[j] for j in range(n_stations)) + \
                         quicksum(transport_costs[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_stations)) + \
                         quicksum(item_profits[i] * knapsack_vars[(i, j)] for i in range(number_of_items) for j in range(n_stations))

        model.setObjective(objective_expr, "minimize")

        for i in range(n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(n_stations)) == demands[i], f"Demand_{i}")

        for j in range(n_stations):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) <= station_capacities[j] * open_stations[j], f"StationCapacity_{j}")

        for k in range(n_renewables):
            model.addCons(renewable_supply[k] <= renewable_capacities[k], f"RenewableCapacity_{k}")

        for j in range(n_stations):
            model.addCons(quicksum(renewable_supply[k] for k in range(n_renewables)) >= quicksum(flow[i, j] for i in range(n_customers)) * open_stations[j], f"RenewableSupplyLink_{j}")

        for i in range(number_of_items):
            model.addCons(quicksum(knapsack_vars[(i, j)] for j in range(n_stations)) <= 1, f"ItemAssignment_{i}")

        for j in range(n_stations):
            model.addCons(quicksum(item_weights[i] * knapsack_vars[(i, j)] for i in range(number_of_items)) <= station_capacities[j], f"KnapsackCapacity_{j}")

        for e_item in essential_items:
            model.addCons(quicksum(knapsack_vars[(e_item, j)] for j in range(n_stations)) >= 1, f"EssentialItemCover_{e_item}")

        specific_station, specific_item = 0, 2
        model.addCons(open_stations[specific_station] == knapsack_vars[(specific_item, specific_station)], f"LogicalCondition_ItemPlacement_{specific_item}_{specific_station}")

        for j in range(n_stations):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) * open_stations[j] <= quicksum(renewable_supply[k] for k in range(n_renewables)), f"LogicalCondition_StationRenewable_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 25,
        'n_stations': 600,
        'n_renewables': 30,
        'demand_interval': (560, 2800),
        'station_capacity_interval': (525, 2100),
        'renewable_capacity_scale': 750.0,
        'fixed_cost_interval': (131, 525),
        'transport_cost_scale': 810.0,
        'number_of_items': 200,
        'number_of_essential_items': 10,
    }

    ev_charging_optimization = EVChargingStationOptimization(parameters, seed=seed)
    instance = ev_charging_optimization.generate_instance()
    solve_status, solve_time = ev_charging_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")