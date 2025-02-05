import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class VehicleRoutingProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # Generate distances between depots and customers
        distances = np.random.randint(self.min_distance, self.max_distance, (self.n_vehicles, self.n_customers))
        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost, self.n_vehicles)
        emission_levels = np.random.randint(self.min_emission, self.max_emission, (self.n_vehicles, self.n_customers))
        customer_effort_levels = np.random.randint(self.min_effort, self.max_effort, self.n_customers)
        vehicle_capacity = np.random.randint(self.min_capacity, self.max_capacity, size=self.n_vehicles)

        res = {
            'distances': distances,
            'vehicle_costs': vehicle_costs,
            'emission_levels': emission_levels,
            'customer_effort_levels': customer_effort_levels,
            'vehicle_capacity': vehicle_capacity
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        distances = instance['distances']
        vehicle_costs = instance['vehicle_costs']
        emission_levels = instance['emission_levels']
        customer_effort_levels = instance['customer_effort_levels']
        vehicle_capacity = instance['vehicle_capacity']

        model = Model("VehicleRoutingProblem")
        vehicle_routes = {}
        vehicle_usage = {}
        emission_exceeded = {}

        # Create variables and set objective for vehicle routing
        for i in range(self.n_vehicles):
            vehicle_usage[i] = model.addVar(vtype="B", name=f"v_{i}", obj=vehicle_costs[i])
            emission_exceeded[i] = model.addVar(vtype="B", name=f"z_{i}", obj=self.emission_penalty)

            for j in range(self.n_customers):
                vehicle_routes[i, j] = model.addVar(vtype="B", name=f"r_{i}_{j}", obj=distances[i, j])

        # Add constraints to ensure each customer is visited by exactly one vehicle
        for j in range(self.n_customers):
            model.addCons(quicksum(vehicle_routes[i, j] for i in range(self.n_vehicles)) == 1, f"Visit_Customer_{j}")

        # Ensure total mileage constraints per vehicle
        for i in range(self.n_vehicles):
            model.addCons(quicksum(vehicle_routes[i, j] * distances[i, j] for j in range(self.n_customers)) <= vehicle_capacity[i], f"Vehicle_Capacity_{i}")

        # Add emission constraints
        for i in range(self.n_vehicles):
            model.addCons(quicksum(vehicle_routes[i, j] * emission_levels[i, j] for j in range(self.n_customers)) <= self.emission_limit + emission_exceeded[i] * self.emission_penalty, f"Emission_Limit_{i}")

        # Ensure every customer is serviced with necessary effort levels
        for j in range(self.n_customers):
            model.addCons(quicksum(vehicle_routes[i, j] * customer_effort_levels[j] for i in range(self.n_vehicles)) <= self.max_effort, f"Effort_Requirement_{j}")

        # Objective: Minimize total cost including vehicle usage costs, routing costs, and emission surpassing penalties
        total_vehicle_cost = quicksum(vehicle_usage[i] * vehicle_costs[i] for i in range(self.n_vehicles))
        routing_cost = quicksum(vehicle_routes[i, j] * distances[i, j] for i in range(self.n_vehicles) for j in range(self.n_customers))
        emission_penalties = quicksum(emission_exceeded[i] * self.emission_penalty for i in range(self.n_vehicles))

        model.setObjective(total_vehicle_cost + routing_cost + emission_penalties, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vehicles': 20,
        'n_customers': 150,
        'min_distance': 5,
        'max_distance': 200,
        'min_vehicle_cost': 3000,
        'max_vehicle_cost': 5000,
        'min_emission': 90,
        'max_emission': 1600,
        'emission_penalty': 2700,
        'emission_limit': 2400,
        'min_effort': 35,
        'max_effort': 60,
        'max_effort_total': 1000,
        'min_capacity': 250,
        'max_capacity': 1400,
    }

    problem = VehicleRoutingProblem(parameters, seed=seed)
    instance = problem.generate_instance()
    solve_status, solve_time = problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")