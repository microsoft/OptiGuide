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

    def efficient_greedy_clique_partition(self):
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

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
        assert self.Number_of_Vessels > 0 and self.Number_of_Routes > 0
        assert self.Min_Vessel_Cost >= 0 and self.Max_Vessel_Cost >= self.Min_Vessel_Cost
        assert self.Route_Cost_Lower_Bound >= 0 and self.Route_Cost_Upper_Bound >= self.Route_Cost_Lower_Bound

        vessel_costs = np.random.randint(self.Min_Vessel_Cost, self.Max_Vessel_Cost + 1, self.Number_of_Vessels)
        route_costs = np.random.randint(self.Route_Cost_Lower_Bound, self.Route_Cost_Upper_Bound + 1, (self.Number_of_Vessels, self.Number_of_Routes))
        route_demands = np.random.randint(1, 10, self.Number_of_Routes)
        
        compatible_fuel_costs = np.random.randint(self.Min_Fuel_Cost, self.Max_Fuel_Cost + 1, (self.Number_of_Vessels, self.Number_of_Fuel_Types))
        weather_impacts = np.random.randint(0, 2, (self.Number_of_Vessels, self.Number_of_Routes))
        emission_reductions = np.random.uniform(self.Min_Emission_Reduction, self.Max_Emission_Reduction, self.Number_of_Vessels)
        
        graph = Graph.barabasi_albert(self.Number_of_Vessels, self.Affinity)
        cliques = graph.efficient_greedy_clique_partition()
        incompatibilities = set(graph.edges)
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                incompatibilities.remove(edge)
            if len(clique) > 1:
                incompatibilities.add(clique)

        used_nodes = set()
        for group in incompatibilities:
            used_nodes.update(group)
        for node in range(self.Number_of_Vessels):
            if node not in used_nodes:
                incompatibilities.add((node,))
        
        return {
            "vessel_costs": vessel_costs,
            "route_costs": route_costs,
            "route_demands": route_demands,
            "compatible_fuel_costs": compatible_fuel_costs,
            "weather_impacts": weather_impacts,
            "emission_reductions": emission_reductions,
            "graph": graph,
            "incompatibilities": incompatibilities,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        vessel_costs = instance['vessel_costs']
        route_costs = instance['route_costs']
        route_demands = instance['route_demands']
        compatible_fuel_costs = instance['compatible_fuel_costs']
        weather_impacts = instance['weather_impacts']
        emission_reductions = instance['emission_reductions']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        
        model = Model("FleetManagement")
        number_of_vessels = len(vessel_costs)
        number_of_routes = len(route_costs[0])
        number_of_fuel_types = len(compatible_fuel_costs[0])

        M = 10e6  # Big M for constraint relaxations
        
        # Decision variables
        vessel_vars = {v: model.addVar(vtype="B", name=f"Vessel_{v}") for v in range(number_of_vessels)}
        route_vars = {(v, r): model.addVar(vtype="B", name=f"Vessel_{v}_Route_{r}") for v in range(number_of_vessels) for r in range(number_of_routes)}
        fuel_vars = {(v, f): model.addVar(vtype="B", name=f"Vessel_{v}_Fuel_{f}") for v in range(number_of_vessels) for f in range(number_of_fuel_types)}
        emission_vars = {v: model.addVar(vtype="C", name=f"Emission_{v}", lb=0) for v in range(number_of_vessels)}
        
        # Objective: minimize the total cost including vessel startup costs, route service costs, and fuel costs
        model.setObjective(
            quicksum(vessel_costs[v] * vessel_vars[v] for v in range(number_of_vessels)) +
            quicksum(route_costs[v, r] * route_vars[v, r] for v in range(number_of_vessels) for r in range(number_of_routes)) +
            quicksum(compatible_fuel_costs[v, f] * fuel_vars[v, f] for v in range(number_of_vessels) for f in range(number_of_fuel_types)), "minimize"
        )
        
        # Constraints: Each route must be serviced by exactly one vessel
        for r in range(number_of_routes):
            model.addCons(quicksum(route_vars[v, r] for v in range(number_of_vessels)) == 1, f"Route_{r}_Demand")
        
        # Constraints: Only active vessels can serve routes
        for v in range(number_of_vessels):
            for r in range(number_of_routes):
                model.addCons(route_vars[v, r] <= vessel_vars[v], f"Vessel_{v}_Serve_{r}")
        
        # Constraints: Emission reduction targets
        for v in range(number_of_vessels):
            model.addCons(emission_vars[v] >= emission_reductions[v], f"Vessel_{v}_Emission_Target")
        
        # Constraints: Weather impacts
        for v in range(number_of_vessels):
            for r in range(number_of_routes):
                model.addCons(route_vars[v, r] <= weather_impacts[v, r], f"Weather_Impact_Vessel_{v}_{r}")

        # Constraints: Fuel type selection
        for v in range(number_of_vessels):
            model.addCons(quicksum(fuel_vars[v, f] for f in range(number_of_fuel_types)) == 1, f"Fuel_Type_Vessel_{v}")

        # Constraints: Truck Graph Incompatibilities
        for count, group in enumerate(incompatibilities):
            if isinstance(group, tuple):
                model.addCons(quicksum(vessel_vars[node] for node in group) <= len(group) - 1, f"Incompatibility_{count}")
            else:
                model.addCons(vessel_vars[group] <= 1, f"Incompatibility_{count}")

        # Additional Clique-Based Constraints
        for count, clique in enumerate(graph.efficient_greedy_clique_partition()):
            if len(clique) > 1:
                model.addCons(quicksum(vessel_vars[node] for node in clique) <= 1, f"Clique_{count}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    parameters = {
        'Number_of_Vessels': 100,
        'Number_of_Routes': 60,
        'Number_of_Fuel_Types': 9,
        'Route_Cost_Lower_Bound': 1500,
        'Route_Cost_Upper_Bound': 5000,
        'Min_Vessel_Cost': 3000,
        'Max_Vessel_Cost': 10000,
        'Min_Fuel_Cost': 350,
        'Max_Fuel_Cost': 2000,
        'Min_Emission_Reduction': 700,
        'Max_Emission_Reduction': 2500,
        'Affinity': 3,
    }
    seed = 42

    fleet_management_optimizer = FleetManagement(parameters, seed=seed)
    instance = fleet_management_optimizer.generate_instance()
    solve_status, solve_time, objective_value = fleet_management_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")