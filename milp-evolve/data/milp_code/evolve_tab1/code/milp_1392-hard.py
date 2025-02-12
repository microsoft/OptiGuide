import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict


class CapacitatedPickupAndDelivery:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers, 1) - rand(1, self.n_hubs))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_hubs))**2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_hubs, self.capacity_interval)
        fixed_costs = self.fixed_cost * np.ones(self.n_hubs)
        transportation_costs = self.generate_transportation_costs() * demands[:, np.newaxis]
        truck_trip_limits = self.randint(self.n_trucks, self.trip_interval)

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'truck_trip_limits': truck_trip_limits
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        truck_trip_limits = instance['truck_trip_limits']

        n_customers = len(demands)
        n_hubs = len(capacities)
        n_trucks = len(truck_trip_limits)
        
        model = Model("CapacitatedPickupAndDelivery")
        
        # Decision variables
        open_hubs = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_hubs)}
        pickup = {(i, j): model.addVar(vtype="B", name=f"Pickup_{i}_{j}") for i in range(n_customers) for j in range(n_hubs)}
        return_trip = {(j, t): model.addVar(vtype="B", name=f"ReturnTrip_{j}_{t}") for j in range(n_hubs) for t in range(n_trucks)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_hubs[j] for j in range(n_hubs)) + quicksum(transportation_costs[i, j] * pickup[i, j] for i in range(n_customers) for j in range(n_hubs))

        # Constraints: customer must be assigned to at least one hub
        for i in range(n_customers):
            model.addCons(quicksum(pickup[i, j] for j in range(n_hubs)) >= 1, f"CustomerPickup_{i}")

        # Constraints: hub capacity limits
        for j in range(n_hubs):
            capacity_expr = quicksum(pickup[i, j] * demands[i] for i in range(n_customers))
            model.addCons(capacity_expr <= capacities[j] * open_hubs[j], f"HubCapacity_{j}")
        
        # Constraints: truck trip limits
        for t in range(n_trucks):
            for j in range(n_hubs):
                trip_expr = quicksum(return_trip[j, t] for j in range(n_hubs))
                model.addCons(trip_expr <= truck_trip_limits[t], f"ReturnTrips_{t}_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 3000,
        'n_hubs': 10,
        'demand_interval': (0, 2),
        'capacity_interval': (13, 216),
        'fixed_cost': 1500,
        'ratio': 393.75,
        'n_trucks': 30,
        'trip_interval': (7, 37),
    }

    delivery_problem = CapacitatedPickupAndDelivery(parameters, seed=seed)
    instance = delivery_problem.generate_instance()
    solve_status, solve_time = delivery_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")