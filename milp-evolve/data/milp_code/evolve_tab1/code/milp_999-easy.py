import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NeighborhoodRevitalization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def transportation_costs(self):
        base_cost = 15.0  # base transportation cost in dollars
        return base_cost * np.random.rand(self.n_neighborhoods, self.n_centers)

    def generate_instance(self):
        resource_availability = self.randint(self.n_centers, self.resource_interval)
        neighborhood_demands = self.randint(self.n_neighborhoods, self.demand_interval)
        activation_costs = self.randint(self.n_centers, self.activation_cost_interval)
        transportation_costs = self.transportation_costs()
        
        collision_avoidance_limits = self.randint(self.n_centers, self.col_avoidance_limit_interval)
        collision_avoidance_cost = self.randint(self.n_centers, self.col_avoidance_cost_interval)
        
        delivery_schedules = self.randint(self.n_neighborhoods, self.schedule_interval)

        res = {
            'resource_availability': resource_availability,
            'neighborhood_demands': neighborhood_demands,
            'activation_costs': activation_costs,
            'transportation_costs': transportation_costs,
            'collision_avoidance_limits': collision_avoidance_limits,
            'collision_avoidance_cost': collision_avoidance_cost,
            'delivery_schedules': delivery_schedules,
        }
        return res

    def solve(self, instance):
        # Instance data
        resource_availability = instance['resource_availability']
        neighborhood_demands = instance['neighborhood_demands']
        activation_costs = instance['activation_costs']
        transportation_costs = instance['transportation_costs']
        collision_avoidance_limits = instance['collision_avoidance_limits']
        collision_avoidance_cost = instance['collision_avoidance_cost']
        delivery_schedules = instance['delivery_schedules']

        n_neighborhoods = len(neighborhood_demands)
        n_centers = len(resource_availability)

        model = Model("NeighborhoodRevitalization")

        # Decision variables
        activate_center = {m: model.addVar(vtype="B", name=f"Activate_{m}") for m in range(n_centers)}
        allocate_resources = {(n, m): model.addVar(vtype="B", name=f"Allocate_{n}_{m}") for n in range(n_neighborhoods) for m in range(n_centers)}
        avoid_collision = {m: model.addVar(vtype="B", name=f"CollisionAvoidance_{m}") for m in range(n_centers)}

        # Schedule variables
        schedule_time = {(n, m): model.addVar(vtype="C", name=f"ScheduleTime_{n}_{m}") for n in range(n_neighborhoods) for m in range(n_centers)}

        # Objective: Minimize total cost, including activation, transportation, and collision avoidance
        penalty_per_transportation = 50
        
        objective_expr = quicksum(activation_costs[m] * activate_center[m] for m in range(n_centers)) + \
                         penalty_per_transportation * quicksum(transportation_costs[n, m] * allocate_resources[n, m] for n in range(n_neighborhoods) for m in range(n_centers)) + \
                         quicksum(collision_avoidance_cost[m] * avoid_collision[m] for m in range(n_centers))

        # Constraints: each neighborhood demand must be met
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocate_resources[n, m] for m in range(n_centers)) == 1, f"Neighborhood_Demand_{n}")

        # Constraints: resource availability limits must be respected
        for m in range(n_centers):
            model.addCons(quicksum(neighborhood_demands[n] * allocate_resources[n, m] for n in range(n_neighborhoods)) <= resource_availability[m] * activate_center[m], f"Resource_Availability_{m}")

        # Constraint: Allocation within the schedule limits
        for n in range(n_neighborhoods):
            for m in range(n_centers):
                model.addCons(transportation_costs[n, m] * allocate_resources[n, m] <= activate_center[m] * 150, f"Transportation_Time_Limit_{n}_{m}")

        # Collision avoidance within limits
        for m in range(n_centers):
            model.addCons(avoid_collision[m] <= collision_avoidance_limits[m], f"Collision_Avoidance_Limit_{m}")

        # Schedule must fit within delivery time
        for n in range(n_neighborhoods):
            for m in range(n_centers):
                model.addCons(schedule_time[n, m] == delivery_schedules[n], f"Delivery_Schedule_{n}_{m}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_neighborhoods': 22,
        'n_centers': 600,
        'resource_interval': (300, 1500),
        'demand_interval': (75, 375),
        'activation_cost_interval': (1000, 3000),
        'col_avoidance_limit_interval': (40, 140),
        'col_avoidance_cost_interval': (700, 2100),
        'schedule_interval': (112, 450),
    }
    
    neighborhood_revitalization = NeighborhoodRevitalization(parameters, seed=seed)
    instance = neighborhood_revitalization.generate_instance()
    solve_status, solve_time = neighborhood_revitalization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")