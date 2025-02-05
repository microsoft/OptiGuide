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

class ResidentialZoneDevelopment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_markets > 0 and self.n_zones > 0
        assert self.min_market_cost >= 0 and self.max_market_cost >= self.min_market_cost
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_market_cap > 0 and self.max_market_cap >= self.min_market_cap

        market_costs = np.random.randint(self.min_market_cost, self.max_market_cost + 1, self.n_markets)
        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, (self.n_markets, self.n_zones))
        capacities = np.random.randint(self.min_market_cap, self.max_market_cap + 1, self.n_markets)
        demands = np.random.randint(1, 10, self.n_zones)

        graph = Graph.barabasi_albert(self.n_markets, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        zone_constraints = set(graph.edges)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                zone_constraints.remove(edge)
            if len(clique) > 1:
                zone_constraints.add(clique)

        used_nodes = set()
        for group in zone_constraints:
            used_nodes.update(group)
        for node in range(self.n_markets):
            if node not in used_nodes:
                zone_constraints.add((node,))
        
        return {
            "market_costs": market_costs,
            "zone_costs": zone_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "zone_constraints": zone_constraints,
            "edge_weights": edge_weights,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        market_costs = instance['market_costs']
        zone_costs = instance['zone_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        zone_constraints = instance['zone_constraints']
        edge_weights = instance['edge_weights']
        
        model = Model("ResidentialZoneDevelopment")
        n_markets = len(market_costs)
        n_zones = len(zone_costs[0])
        
        # Decision variables
        market_vars = {m: model.addVar(vtype="B", name=f"Market_{m}") for m in range(n_markets)}
        zone_vars = {(m, z): model.addVar(vtype="B", name=f"Market_{m}_Zone_{z}") for m in range(n_markets) for z in range(n_zones)}
        zone_edge_vars = {edge: model.addVar(vtype="B", name=f"ZoneEdge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        
        # Objective: minimize the total cost including market costs and zone costs
        model.setObjective(
            quicksum(market_costs[m] * market_vars[m] for m in range(n_markets)) +
            quicksum(zone_costs[m, z] * zone_vars[m, z] for m in range(n_markets) for z in range(n_zones)) +
            quicksum(edge_weights[i] * zone_edge_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )
        
        # Constraints: Each zone demand is met by exactly one market
        for z in range(n_zones):
            model.addCons(quicksum(zone_vars[m, z] for m in range(n_markets)) == 1, f"Zone_{z}_Demand")
        
        # Constraints: Only open markets can serve zones
        for m in range(n_markets):
            for z in range(n_zones):
                model.addCons(zone_vars[m, z] <= market_vars[m], f"Market_{m}_Serve_{z}")
        
        # Constraints: Markets cannot exceed their capacities
        for m in range(n_markets):
            model.addCons(quicksum(demands[z] * zone_vars[m, z] for z in range(n_zones)) <= capacities[m], f"Market_{m}_Capacity")
        
        # Constraints: Market Graph Cliques
        for count, group in enumerate(zone_constraints):
            model.addCons(quicksum(market_vars[node] for node in group) <= 1, f"ZoneClique_{count}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_markets': 74,
        'n_zones': 112,
        'min_zone_cost': 36,
        'max_zone_cost': 3000,
        'min_market_cost': 421,
        'max_market_cost': 5000,
        'min_market_cap': 370,
        'max_market_cap': 1260,
        'affinity': 22,
    }

    zone_development_optimizer = ResidentialZoneDevelopment(parameters, seed=42)
    instance = zone_development_optimizer.generate_instance()
    solve_status, solve_time, objective_value = zone_development_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")