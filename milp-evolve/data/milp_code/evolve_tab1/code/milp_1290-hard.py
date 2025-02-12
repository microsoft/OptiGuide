import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class FleetManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.Number_of_Trucks > 0 and self.Number_of_Routes > 0
        assert self.Min_Truck_Cost >= 0 and self.Max_Truck_Cost >= self.Min_Truck_Cost
        assert self.Route_Cost_Lower_Bound >= 0 and self.Route_Cost_Upper_Bound >= self.Route_Cost_Lower_Bound
        assert self.Min_Truck_Capacity > 0 and self.Max_Truck_Capacity >= self.Min_Truck_Capacity

        truck_costs = np.random.randint(self.Min_Truck_Cost, self.Max_Truck_Cost + 1, self.Number_of_Trucks)
        route_costs = np.random.randint(self.Route_Cost_Lower_Bound, self.Route_Cost_Upper_Bound + 1, (self.Number_of_Trucks, self.Number_of_Routes))
        truck_capacities = np.random.randint(self.Min_Truck_Capacity, self.Max_Truck_Capacity + 1, self.Number_of_Trucks)

        # Variable route demands
        route_demands = np.random.poisson(self.Average_Demand, self.Number_of_Routes)
        
        graph = Graph.barabasi_albert(self.Number_of_Trucks, self.Affinity)
        incompatibilities = set(graph.edges)

        # Temperature-sensitive supplies and related truck capabilities
        temperature_sensitive_routes = np.random.choice([0, 1], self.Number_of_Routes, p=[0.8, 0.2])
        refrigerated_trucks = np.random.choice([0, 1], self.Number_of_Trucks, p=[0.6, 0.4])

        # Generate time slots for traffic congestion
        time_slots = np.random.randint(1, self.Time_Slots + 1, self.Number_of_Routes)

        return {
            "truck_costs": truck_costs,
            "route_costs": route_costs,
            "truck_capacities": truck_capacities,
            "route_demands": route_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
            "temperature_sensitive_routes": temperature_sensitive_routes,
            "refrigerated_trucks": refrigerated_trucks,
            "time_slots": time_slots,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        truck_costs = instance['truck_costs']
        route_costs = instance['route_costs']
        truck_capacities = instance['truck_capacities']
        route_demands = instance['route_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        temperature_sensitive_routes = instance['temperature_sensitive_routes']
        refrigerated_trucks = instance['refrigerated_trucks']
        time_slots = instance['time_slots']
        
        model = Model("FleetManagement")
        number_of_trucks = len(truck_costs)
        number_of_routes = len(route_costs[0])

        M = sum(truck_capacities)  # Big M
        
        # Decision variables
        truck_vars = {t: model.addVar(vtype="B", name=f"Truck_{t}") for t in range(number_of_trucks)}
        route_vars = {(t, r): model.addVar(vtype="B", name=f"Truck_{t}_Route_{r}") for t in range(number_of_trucks) for r in range(number_of_routes)}
        
        # Objective: minimize the total cost including truck startup costs and route service costs
        model.setObjective(
            quicksum(truck_costs[t] * truck_vars[t] for t in range(number_of_trucks)) +
            quicksum(route_costs[t, r] * route_vars[t, r] for t in range(number_of_trucks) for r in range(number_of_routes)), "minimize"
        )
        
        # Constraints: Each route must be served by exactly one truck
        for r in range(number_of_routes):
            model.addCons(quicksum(route_vars[t, r] for t in range(number_of_trucks)) == 1, f"Route_{r}_Demand")
        
        # Constraints: Only active trucks can serve routes
        for t in range(number_of_trucks):
            for r in range(number_of_routes):
                model.addCons(route_vars[t, r] <= truck_vars[t], f"Truck_{t}_Serve_{r}")
        
        # Constraints: Trucks cannot exceed their capacity using Big M
        for t in range(number_of_trucks):
            model.addCons(quicksum(route_demands[r] * route_vars[t, r] for r in range(number_of_routes)) <= truck_capacities[t] * truck_vars[t], f"Truck_{t}_Capacity")

        # Constraints: Truck Graph Incompatibilities
        for count, (i, j) in enumerate(incompatibilities):
            model.addCons(truck_vars[i] + truck_vars[j] <= 1, f"Incompatibility_{count}")
        
        # Constraint: Temperature-sensitive supplies must be handled by refrigerated trucks only
        for t in range(number_of_trucks):
            for r in range(number_of_routes):
                if temperature_sensitive_routes[r]:
                    model.addCons(route_vars[t, r] <= refrigerated_trucks[t], f"Temp_Sensitive_{t}_{r}")

        # Constraint: Add traffic congestion impact
        for t in range(number_of_trucks):
            for ts in range(1, self.Time_Slots + 1):
                model.addCons(quicksum(route_vars[t, r] for r in range(number_of_routes) if time_slots[r] == ts) <= self.Max_Routes_Per_Time_Slot, f"Traffic_{t}_{ts}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Trucks': 120,
        'Number_of_Routes': 90,
        'Route_Cost_Lower_Bound': 200,
        'Route_Cost_Upper_Bound': 3000,
        'Min_Truck_Cost': 2400,
        'Max_Truck_Cost': 5000,
        'Min_Truck_Capacity': 1600,
        'Max_Truck_Capacity': 1600,
        'Affinity': 4,
        'Average_Demand': 3,
        'Time_Slots': 2,
        'Max_Routes_Per_Time_Slot': 30,
    }

    fleet_management_optimizer = FleetManagement(parameters, seed=42)
    instance = fleet_management_optimizer.generate_instance()
    solve_status, solve_time, objective_value = fleet_management_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")