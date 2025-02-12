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
    
    # Data Generation
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

        res = {
            'demands': demands,
            'station_capacities': station_capacities,
            'renewable_capacities': renewable_capacities,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs
        }
        
        # New instance data
        item_weights = self.randint(self.number_of_items, (1, 10))
        item_profits = self.randint(self.number_of_items, (10, 100))
        knapsack_capacities = self.randint(self.number_of_knapsacks, (30, 100))
        essential_items = random.sample(range(self.number_of_items), self.number_of_essential_items)
        
        res.update({
            'item_weights': item_weights,
            'item_profits': item_profits,
            'knapsack_capacities': knapsack_capacities,
            'essential_items': essential_items
        })
        return res

    # MILP Solver
    def solve(self, instance):
        demands = instance['demands']
        station_capacities = instance['station_capacities']
        renewable_capacities = instance['renewable_capacities']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        item_weights = instance['item_weights']
        item_profits = instance['item_profits']
        knapsack_capacities = instance['knapsack_capacities']
        essential_items = instance['essential_items']
        
        n_customers = len(demands)
        n_stations = len(station_capacities)
        n_renewables = len(renewable_capacities)
        number_of_items = len(item_weights)
        number_of_knapsacks = len(knapsack_capacities)
        
        model = Model("EVChargingStationOptimization")
        
        # Decision variables
        open_stations = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_stations)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(n_customers) for j in range(n_stations)}
        renewable_supply = {j: model.addVar(vtype="C", name=f"RenewableSupply_{j}") for j in range(n_renewables)}
        knapsack_vars = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(number_of_items) for j in range(number_of_knapsacks)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_stations[j] for j in range(n_stations)) + \
                         quicksum(transport_costs[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_stations)) + \
                         quicksum(item_profits[i] * knapsack_vars[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))

        model.setObjective(objective_expr, "minimize")

        # Demand satisfaction constraints
        for i in range(n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(n_stations)) == demands[i], f"Demand_{i}")
        
        # Station capacity constraints
        for j in range(n_stations):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) <= station_capacities[j] * open_stations[j], f"StationCapacity_{j}")
        
        # Renewable supply constraints
        for k in range(n_renewables):
            model.addCons(renewable_supply[k] <= renewable_capacities[k], f"RenewableCapacity_{k}")

        # Linking renewable supply to station energy inflow
        for j in range(n_stations):
            model.addCons(quicksum(renewable_supply[k] for k in range(n_renewables)) >= quicksum(flow[i, j] for i in range(n_customers)) * open_stations[j], f"RenewableSupplyLink_{j}")

        # Items in at most one knapsack
        for i in range(number_of_items):
            model.addCons(quicksum(knapsack_vars[(i, j)] for j in range(number_of_knapsacks)) <= 1, f"ItemAssignment_{i}")

        # Knapsack capacity constraints
        for j in range(number_of_knapsacks):
            model.addCons(quicksum(item_weights[i] * knapsack_vars[(i, j)] for i in range(number_of_items)) <= knapsack_capacities[j], f"KnapsackCapacity_{j}")

        # Set covering constraints: essential items must be covered by at least one knapsack
        for e_item in essential_items:
            model.addCons(quicksum(knapsack_vars[(e_item, j)] for j in range(number_of_knapsacks)) >= 1, f"EssentialItemCover_{e_item}")

        # Logical conditions added to enhance complexity
        # Logical Condition 1: If a specific station is open, a specific item must be in a knapsack
        specific_station, specific_item = 0, 2
        model.addCons(open_stations[specific_station] == knapsack_vars[(specific_item, 0)], f"LogicalCondition_ItemPlacement_{specific_item}_{specific_station}")

        # Logical Condition 2: If certain stations are used, renewable supply must be linked logically
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
        'number_of_knapsacks': 150,
        'number_of_essential_items': 10,
    }
    
    ev_charging_optimization = EVChargingStationOptimization(parameters, seed=seed)
    instance = ev_charging_optimization.generate_instance()
    solve_status, solve_time = ev_charging_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")