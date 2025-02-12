import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_stops, routes, degrees, neighbors):
        self.number_of_stops = number_of_stops
        self.stops = np.arange(number_of_stops)
        self.routes = routes
        self.degrees = degrees
        self.neighbors = neighbors

    def efficient_greedy_route_partition(self):
        routes = []
        remaining_stops = (-self.degrees).argsort().tolist()

        while remaining_stops:
            route_center, remaining_stops = remaining_stops[0], remaining_stops[1:]
            route = {route_center}
            neighbors = self.neighbors[route_center].intersection(remaining_stops)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                if all([neighbor in self.neighbors[route_stop] for route_stop in route]):
                    route.add(neighbor)
            routes.append(route)
            remaining_stops = [stop for stop in remaining_stops if stop not in route]

        return routes

    @staticmethod
    def barabasi_albert(number_of_stops, affinity):
        assert affinity >= 1 and affinity < number_of_stops

        routes = set()
        degrees = np.zeros(number_of_stops, dtype=int)
        neighbors = {stop: set() for stop in range(number_of_stops)}
        for new_stop in range(affinity, number_of_stops):
            if new_stop == affinity:
                neighborhood = np.arange(new_stop)
            else:
                neighbor_prob = degrees[:new_stop] / (2 * len(routes))
                neighborhood = np.random.choice(new_stop, affinity, replace=False, p=neighbor_prob)
            for stop in neighborhood:
                routes.add((stop, new_stop))
                degrees[stop] += 1
                degrees[new_stop] += 1
                neighbors[stop].add(new_stop)
                neighbors[new_stop].add(stop)

        graph = Graph(number_of_stops, routes, degrees, neighbors)
        return graph

class PublicTransportOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.Number_of_Buses > 0 and self.Number_of_Zones > 0
        assert self.Min_Route_Cost >= 0 and self.Max_Route_Cost >= self.Min_Route_Cost
        assert self.Zone_Cost_Lower_Bound >= 0 and self.Zone_Cost_Upper_Bound >= self.Zone_Cost_Lower_Bound
        assert self.Min_Capacity > 0 and self.Max_Capacity >= self.Min_Capacity
        assert self.Min_Boarding_Time >= 0 and self.Max_Boarding_Time >= self.Min_Boarding_Time
        assert self.Min_Holiday_Penalty >= 0 and self.Max_Holiday_Penalty >= self.Min_Holiday_Penalty

        route_costs = np.random.randint(self.Min_Route_Cost, self.Max_Route_Cost + 1, self.Number_of_Buses)
        zone_costs = np.random.randint(self.Zone_Cost_Lower_Bound, self.Zone_Cost_Upper_Bound + 1, (self.Number_of_Buses, self.Number_of_Zones))
        bus_capacities = np.random.randint(self.Min_Capacity, self.Max_Capacity + 1, self.Number_of_Buses)
        zone_demands = np.random.randint(1, 10, self.Number_of_Zones)
        boarding_times = np.random.uniform(self.Min_Boarding_Time, self.Max_Boarding_Time, self.Number_of_Buses)
        holiday_penalties = np.random.uniform(self.Min_Holiday_Penalty, self.Max_Holiday_Penalty, self.Number_of_Buses)
        
        graph = Graph.barabasi_albert(self.Number_of_Buses, self.Affinity)
        routes = graph.efficient_greedy_route_partition()
        incompatibilities = set(graph.routes)
        edge_weights = np.random.randint(1, 10, size=len(graph.routes))
        
        for route in routes:
            route = tuple(sorted(route))
            for edge in combinations(route, 2):
                incompatibilities.remove(edge)
            if len(route) > 1:
                incompatibilities.add(route)

        used_stops = set()
        for group in incompatibilities:
            used_stops.update(group)
        for stop in range(self.Number_of_Buses):
            if stop not in used_stops:
                incompatibilities.add((stop,))
        
        return {
            "route_costs": route_costs,
            "zone_costs": zone_costs,
            "bus_capacities": bus_capacities,
            "zone_demands": zone_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
            "edge_weights": edge_weights,
            "boarding_times": boarding_times,
            "holiday_penalties": holiday_penalties,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        route_costs = instance['route_costs']
        zone_costs = instance['zone_costs']
        bus_capacities = instance['bus_capacities']
        zone_demands = instance['zone_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        edge_weights = instance['edge_weights']
        boarding_times = instance['boarding_times']
        holiday_penalties = instance['holiday_penalties']
        
        model = Model("PublicTransportOptimization")
        number_of_buses = len(route_costs)
        number_of_zones = len(zone_costs[0])
        
        # Decision Variables
        bus_vars = {b: model.addVar(vtype="B", name=f"Bus_{b}") for b in range(number_of_buses)}
        zone_vars = {(b, z): model.addVar(vtype="B", name=f"Bus_{b}_Zone_{z}") for b in range(number_of_buses) for z in range(number_of_zones)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.routes}
        boarding_time_vars = {b: model.addVar(vtype="C", lb=0, name=f"BusBoardingTime_{b}") for b in range(number_of_buses)}
        holiday_penalty_vars = {b: model.addVar(vtype="C", lb=0, name=f"BusHolidayPenalty_{b}") for b in range(number_of_buses)}

        # Objective: minimize the total cost including the penalties for holiday operations and route operating costs
        model.setObjective(
            quicksum(route_costs[b] * bus_vars[b] for b in range(number_of_buses)) +
            quicksum(zone_costs[b, z] * zone_vars[b, z] for b in range(number_of_buses) for z in range(number_of_zones)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.routes)) +
            quicksum(boarding_time_vars[b] for b in range(number_of_buses)) +
            quicksum(holiday_penalty_vars[b] for b in range(number_of_buses)), "minimize"
        )
        
        # Constraints: Each zone must be served by exactly one bus
        for z in range(number_of_zones):
            model.addCons(quicksum(zone_vars[b, z] for b in range(number_of_buses)) == 1, f"Zone_{z}_Service")
        
        # Constraints: Only operating buses can serve zones
        for b in range(number_of_buses):
            for z in range(number_of_zones):
                model.addCons(zone_vars[b, z] <= bus_vars[b], f"Bus_{b}_Service_{z}")
        
        # Constraints: Bus capacities cannot be exceeded by zones' demands
        for b in range(number_of_buses):
            model.addCons(quicksum(zone_demands[z] * zone_vars[b, z] for z in range(number_of_zones)) <= bus_capacities[b], f"Bus_{b}_Capacity")
        
        # Constraints: Service Incompatibilities
        for count, group in enumerate(incompatibilities):
            model.addCons(quicksum(bus_vars[stop] for stop in group) <= 1, f"Incompatibility_{count}")

        # Compatibility Constraints: Prohibit incompatible buses from serving the same zones
        for i, neighbors in graph.neighbors.items():
            for neighbor in neighbors:
                model.addCons(bus_vars[i] + bus_vars[neighbor] <= 1, f"Neighbor_{i}_{neighbor}")

        # Boarding time constraints during normal operations
        for b in range(number_of_buses):
            model.addCons(boarding_time_vars[b] == boarding_times[b] * bus_capacities[b] * bus_vars[b], f"Bus_{b}_BoardingTime")
        
        # Holiday penalty constraints for exceeding boarding times
        for b in range(number_of_buses):
            model.addCons(holiday_penalty_vars[b] == holiday_penalties[b] * bus_capacities[b] * bus_vars[b], f"Bus_{b}_HolidayPenalty")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Buses': 100,
        'Number_of_Zones': 100,
        'Zone_Cost_Lower_Bound': 2,
        'Zone_Cost_Upper_Bound': 2500,
        'Min_Route_Cost': 1000,
        'Max_Route_Cost': 4500,
        'Min_Capacity': 900,
        'Max_Capacity': 1800,
        'Affinity': 40,
        'Min_Boarding_Time': 0.31,
        'Max_Boarding_Time': 0.31,
        'Min_Holiday_Penalty': 0.31,
        'Max_Holiday_Penalty': 0.73,
    }
    
    transport_optimizer = PublicTransportOptimization(parameters, seed=42)
    instance = transport_optimizer.generate_instance()
    solve_status, solve_time, objective_value = transport_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")