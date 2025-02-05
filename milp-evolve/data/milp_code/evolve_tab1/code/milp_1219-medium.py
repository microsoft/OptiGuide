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

class NetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_hubs > 0 and self.n_nodes > 0
        assert self.min_hub_cost >= 0 and self.max_hub_cost >= self.min_hub_cost
        assert self.min_link_cost >= 0 and self.max_link_cost >= self.min_link_cost
        assert self.min_hub_capacity > 0 and self.max_hub_capacity >= self.min_hub_capacity

        hub_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.n_hubs)
        link_costs = np.random.randint(self.min_link_cost, self.max_link_cost + 1, (self.n_hubs, self.n_nodes))
        capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.n_hubs)
        demands = np.random.randint(10, 100, self.n_nodes)

        graph = Graph.barabasi_albert(self.n_hubs, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))
        
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
            "link_costs": link_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_weights": edge_weights,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hub_costs = instance['hub_costs']
        link_costs = instance['link_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_weights = instance['edge_weights']
        
        model = Model("NetworkOptimization")
        n_hubs = len(hub_costs)
        n_nodes = len(link_costs[0])
        
        # Decision variables
        hub_vars = {h: model.addVar(vtype="B", name=f"Hub_{h}") for h in range(n_hubs)}
        node_vars = {(h, n): model.addVar(vtype="B", name=f"Hub_{h}_Node_{n}") for h in range(n_hubs) for n in range(n_nodes)}
        capacity_vars = {edge: model.addVar(vtype="B", name=f"Capacity_{edge[0]}_{edge[1]}") for edge in graph.edges}
        
        # Objective: minimize the total cost including hub costs and link costs
        model.setObjective(
            quicksum(hub_costs[h] * hub_vars[h] for h in range(n_hubs)) +
            quicksum(link_costs[h, n] * node_vars[h, n] for h in range(n_hubs) for n in range(n_nodes)) +
            quicksum(edge_weights[i] * capacity_vars[edge] for i, edge in enumerate(graph.edges)), 
            "minimize"
        )
        
        # Constraints: Each node demand is met by exactly one hub
        for n in range(n_nodes):
            model.addCons(quicksum(node_vars[h, n] for h in range(n_hubs)) == 1, f"Node_{n}_Demand")

        # Constraints: Only open hubs can serve nodes
        for h in range(n_hubs):
            for n in range(n_nodes):
                model.addCons(node_vars[h, n] <= hub_vars[h], f"Hub_{h}_Serve_{n}")
        
        # Constraints: Hubs cannot exceed their capacity
        for h in range(n_hubs):
            model.addCons(quicksum(demands[n] * node_vars[h, n] for n in range(n_nodes)) <= capacities[h], f"Hub_{h}_Capacity")
        
        M = max(hub_costs) + 1  # Big M constant for the formulation
        # Constraints: Hub Graph Cliques with Big M
        for count, group in enumerate(inequalities):
            if len(group) > 1:
                for u, v in combinations(group, 2):
                    model.addCons(hub_vars[u] + hub_vars[v] <= capacity_vars[(u, v)] + 1, f"CliqueBigM_{u}_{v}")
                    model.addCons(capacity_vars[(u, v)] <= hub_vars[u], f"EdgeBigM_u_{u}_{v}")
                    model.addCons(capacity_vars[(u, v)] <= hub_vars[v], f"EdgeBigM_u_{v}_{u}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hubs': 55,
        'n_nodes': 112,
        'min_link_cost': 6,
        'max_link_cost': 3000,
        'min_hub_cost': 1264,
        'max_hub_cost': 5000,
        'min_hub_capacity': 46,
        'max_hub_capacity': 1416,
        'affinity': 44,
    }

    network_optimizer = NetworkOptimization(parameters, seed=42)
    instance = network_optimizer.generate_instance()
    solve_status, solve_time, objective_value = network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")