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

class LogisticsNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.Number_of_Depots > 0 and self.Number_of_Cities > 0
        assert self.Min_Depot_Cost >= 0 and self.Max_Depot_Cost >= self.Min_Depot_Cost
        assert self.City_Cost_Lower_Bound >= 0 and self.City_Cost_Upper_Bound >= self.City_Cost_Lower_Bound
        assert self.Min_Depot_Capacity > 0 and self.Max_Depot_Capacity >= self.Min_Depot_Capacity

        depot_costs = np.random.randint(self.Min_Depot_Cost, self.Max_Depot_Cost + 1, self.Number_of_Depots)
        city_costs = np.random.randint(self.City_Cost_Lower_Bound, self.City_Cost_Upper_Bound + 1, (self.Number_of_Depots, self.Number_of_Cities))
        depot_capacities = np.random.randint(self.Min_Depot_Capacity, self.Max_Depot_Capacity + 1, self.Number_of_Depots)
        city_demands = np.random.randint(1, 10, self.Number_of_Cities)
        
        graph = Graph.barabasi_albert(self.Number_of_Depots, self.Affinity)
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
        for node in range(self.Number_of_Depots):
            if node not in used_nodes:
                incompatibilities.add((node,))
        
        return {
            "depot_costs": depot_costs,
            "city_costs": city_costs,
            "depot_capacities": depot_capacities,
            "city_demands": city_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        depot_costs = instance['depot_costs']
        city_costs = instance['city_costs']
        depot_capacities = instance['depot_capacities']
        city_demands = instance['city_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        
        model = Model("LogisticsNetworkDesign")
        number_of_depots = len(depot_costs)
        number_of_cities = len(city_costs[0])

        M = sum(depot_capacities)  # Big M
        
        # Decision variables
        depot_vars = {d: model.addVar(vtype="B", name=f"Depot_{d}") for d in range(number_of_depots)}
        city_vars = {(d, c): model.addVar(vtype="B", name=f"Depot_{d}_City_{c}") for d in range(number_of_depots) for c in range(number_of_cities)}
        
        # Objective: minimize the total cost including depot costs and city assignment costs
        model.setObjective(
            quicksum(depot_costs[d] * depot_vars[d] for d in range(number_of_depots)) +
            quicksum(city_costs[d, c] * city_vars[d, c] for d in range(number_of_depots) for c in range(number_of_cities)), "minimize"
        )
        
        # Constraints: Each city demand is met by exactly one depot
        for c in range(number_of_cities):
            model.addCons(quicksum(city_vars[d, c] for d in range(number_of_depots)) == 1, f"City_{c}_Demand")
        
        # Constraints: Only open depots can serve cities
        for d in range(number_of_depots):
            for c in range(number_of_cities):
                model.addCons(city_vars[d, c] <= depot_vars[d], f"Depot_{d}_Serve_{c}")
        
        # Constraints: Depots cannot exceed their capacity using Big M
        for d in range(number_of_depots):
            model.addCons(quicksum(city_demands[c] * city_vars[d, c] for c in range(number_of_cities)) <= depot_capacities[d] * depot_vars[d], f"Depot_{d}_Capacity")

        # Constraints: Depot Graph Incompatibilities
        for count, group in enumerate(incompatibilities):
            model.addCons(quicksum(depot_vars[node] for node in group) <= 1, f"Incompatibility_{count}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Depots': 82,
        'Number_of_Cities': 112,
        'City_Cost_Lower_Bound': 45,
        'City_Cost_Upper_Bound': 3000,
        'Min_Depot_Cost': 1686,
        'Max_Depot_Cost': 5000,
        'Min_Depot_Capacity': 787,
        'Max_Depot_Capacity': 945,
        'Affinity': 11,
    }
    
    logistics_network_optimizer = LogisticsNetworkDesign(parameters, seed=42)
    instance = logistics_network_optimizer.generate_instance()
    solve_status, solve_time, objective_value = logistics_network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")