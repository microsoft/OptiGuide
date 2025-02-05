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

class UrbanTransportationPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_hubs > 0 and self.n_zones > 0
        assert self.min_hub_cost >= 0 and self.max_hub_cost >= self.min_hub_cost
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_hub_capacity > 0 and self.max_hub_capacity >= self.min_hub_capacity

        hub_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.n_hubs)
        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, (self.n_hubs, self.n_zones))
        capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.n_hubs)
        demands = np.random.randint(1, 10, self.n_zones)

        graph = Graph.barabasi_albert(self.n_hubs, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_travel_times = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(self.n_hubs):
            if node not in used_nodes:
                inequalities.add((node,))
        
        return {
            "hub_costs": hub_costs,
            "zone_costs": zone_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_travel_times": edge_travel_times,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hub_costs = instance['hub_costs']
        zone_costs = instance['zone_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_travel_times = instance['edge_travel_times']
        
        model = Model("UrbanTransportationPlanning")
        n_hubs = len(hub_costs)
        n_zones = len(zone_costs[0])
        
        # Decision variables
        hub_vars = {h: model.addVar(vtype="B", name=f"Hub_{h}") for h in range(n_hubs)}
        zone_vars = {(h, z): model.addVar(vtype="B", name=f"Hub_{h}_Zone_{z}") for h in range(n_hubs) for z in range(n_zones)}
        connection_vars = {edge: model.addVar(vtype="B", name=f"Connection_{edge[0]}_{edge[1]}") for edge in graph.edges}
        
        # Objective: minimize the total cost including hub costs, zone servicing costs, and travel times
        model.setObjective(
            quicksum(hub_costs[h] * hub_vars[h] for h in range(n_hubs)) +
            quicksum(zone_costs[h, z] * zone_vars[h, z] for h in range(n_hubs) for z in range(n_zones)) +
            quicksum(edge_travel_times[i] * connection_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )
        
        # Constraints: Each zone is served by exactly one hub
        for z in range(n_zones):
            model.addCons(quicksum(zone_vars[h, z] for h in range(n_hubs)) == 1, f"Zone_{z}_Service")
        
        # Constraints: Only open hubs can serve zones
        for h in range(n_hubs):
            for z in range(n_zones):
                model.addCons(zone_vars[h, z] <= hub_vars[h], f"Hub_{h}_Serve_{z}")
        
        # Constraints: Hubs cannot exceed their capacity
        for h in range(n_hubs):
            model.addCons(quicksum(demands[z] * zone_vars[h, z] for z in range(n_zones)) <= capacities[h], f"Hub_{h}_Capacity")
        
        # Constraints: Hub Graph Cliques
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(hub_vars[node] for node in group) <= 1, f"Clique_{count}")
        
        # Adding complexity: additional clique constraints
        complexities = list(inequalities)  # convert set to list to manipulate
        
        # Additional larger cliques formed from subsets of existing cliques
        # Let's assume the complexity involves adding cliques of size 3 if possible
        new_cliques = []
        for clique in complexities:
            if isinstance(clique, tuple) and len(clique) >= 3:
                new_cliques.extend(list(combinations(clique, 3)))
        
        complexity_vars = {idx: model.addVar(vtype="B", name=f"Complexity_{idx}") for idx in range(len(new_cliques))}
        
        for idx, clique in enumerate(new_cliques):
            model.addCons(quicksum(hub_vars[node] for node in clique) <= complexity_vars[idx], f"Complexity_Clique_{idx}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hubs': 37,
        'n_zones': 112,
        'min_zone_cost': 343,
        'max_zone_cost': 3000,
        'min_hub_cost': 1686,
        'max_hub_cost': 5000,
        'min_hub_capacity': 555,
        'max_hub_capacity': 1260,
        'affinity': 11,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    transportation_optimizer = UrbanTransportationPlanning(parameters, seed)
    instance = transportation_optimizer.generate_instance()
    solve_status, solve_time, objective_value = transportation_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")