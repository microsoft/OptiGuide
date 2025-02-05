import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations
from collections import defaultdict

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
    def erdos_renyi(number_of_nodes, edge_probability):
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        return Graph(number_of_nodes, edges, degrees, neighbors)

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
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)
        return Graph(number_of_nodes, edges, degrees, neighbors)

    @staticmethod
    def watts_strogatz(number_of_nodes, k, beta):
        graph = nx.watts_strogatz_graph(number_of_nodes, k, beta)
        edges = set(graph.edges())
        degrees = np.array([graph.degree(n) for n in range(number_of_nodes)], dtype=int)
        neighbors = {node: set(graph.neighbors(node)) for node in range(number_of_nodes)}
        return Graph(number_of_nodes, edges, degrees, neighbors)


class ExtendedVaccineDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_production_zones > 0 and self.n_distribution_zones > 0
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_production_capacity > 0 and self.max_production_capacity >= self.min_production_capacity

        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, self.n_production_zones)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_production_zones, self.n_distribution_zones))
        production_capacities = np.random.randint(self.min_production_capacity, self.max_production_capacity + 1, self.n_production_zones)
        distribution_demands = np.random.randint(1, 10, self.n_distribution_zones)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.n_production_zones)
        distances = np.random.uniform(0, self.max_transport_distance, (self.n_production_zones, self.n_distribution_zones))
        shipment_costs = np.random.randint(1, 20, (self.n_production_zones, self.n_distribution_zones))
        shipment_capacities = np.random.randint(5, 15, (self.n_production_zones, self.n_distribution_zones))

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.n_production_zones):
            for d in range(self.n_distribution_zones):
                G.add_edge(f"production_{p}", f"distribution_{d}")
                node_pairs.append((f"production_{p}", f"distribution_{d}"))

        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_production_zones, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_production_zones, self.affinity)
        elif self.graph_type == 'watts_strogatz':
            graph = Graph.watts_strogatz(self.n_production_zones, self.ks, self.beta)
        else:
            raise ValueError("Unsupported graph type.")
        
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        distances_between_zones = dict(nx.all_pairs_dijkstra_path_length(nx.Graph(graph.edges)))
        path_constraints = set()
        for i in distances_between_zones:
            for j in distances_between_zones[i]:
                if distances_between_zones[i][j] == 2:
                    path_constraints.add(tuple(sorted((i, j))))

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(self.n_production_zones):
            if node not in used_nodes:
                inequalities.add((node,))
        
        weights = np.random.uniform(1, 10, self.n_production_zones)

        return {
            "zone_costs": zone_costs,
            "transport_costs": transport_costs,
            "production_capacities": production_capacities,
            "distribution_demands": distribution_demands,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "shipment_costs": shipment_costs,
            "shipment_capacities": shipment_capacities,
            "independent_set_inequalities": inequalities,
            "path_constraints": path_constraints,
            "weights": weights,
        }

    def solve(self, instance):
        zone_costs = instance['zone_costs']
        transport_costs = instance['transport_costs']
        production_capacities = instance['production_capacities']
        distribution_demands = instance['distribution_demands']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        shipment_costs = instance['shipment_costs']
        shipment_capacities = instance['shipment_capacities']
        independent_set_inequalities = instance['independent_set_inequalities']
        path_constraints = instance['path_constraints']
        weights = instance['weights']
        
        model = Model("ExtendedVaccineDistributionOptimization")
        n_production_zones = len(zone_costs)
        n_distribution_zones = len(transport_costs[0])

        # Decision variables
        vaccine_vars = {p: model.addVar(vtype="B", name=f"Production_{p}") for p in range(n_production_zones)}
        zone_vars = {(u, v): model.addVar(vtype="C", name=f"Vaccine_{u}_{v}") for u, v in node_pairs}
        shipment_vars = {(u, v): model.addVar(vtype="I", name=f"Shipment_{u}_{v}") for u, v in node_pairs}

        model.setObjective(
            quicksum(zone_costs[p] * vaccine_vars[p] for p in range(n_production_zones)) +
            quicksum(transport_costs[p, int(v.split('_')[1])] * zone_vars[(u, v)] for (u, v) in node_pairs for p in range(n_production_zones) if u == f'production_{p}') +
            quicksum(shipment_costs[int(u.split('_')[1]), int(v.split('_')[1])] * shipment_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(weights[p] * (1 - vaccine_vars[p]) for p in range(n_production_zones)) * 0.2,
            "minimize"
        )

        for d in range(n_distribution_zones):
            model.addCons(
                quicksum(zone_vars[(u, f"distribution_{d}")] for u in G.predecessors(f"distribution_{d}")) == distribution_demands[d], 
                f"Distribution_{d}_NodeFlowConservation"
            )

        for p in range(n_production_zones):
            for d in range(n_distribution_zones):
                model.addCons(
                    zone_vars[(f"production_{p}", f"distribution_{d}")] <= budget_limits[p] * vaccine_vars[p], 
                    f"Production_{p}_VaccineLimitByBudget_{d}"
                )

        for p in range(n_production_zones):
            model.addCons(
                quicksum(zone_vars[(f"production_{p}", f"distribution_{d}")] for d in range(n_distribution_zones)) <= production_capacities[p], 
                f"Production_{p}_MaxZoneCapacity"
            )

        for d in range(n_distribution_zones):
            model.addCons(
                quicksum(vaccine_vars[p] for p in range(n_production_zones) if distances[p, d] <= self.max_transport_distance) >= 1, 
                f"Distribution_{d}_ElderyZoneCoverage"
            )

        for u, v in node_pairs:
            model.addCons(shipment_vars[(u, v)] <= shipment_capacities[int(u.split('_')[1]), int(v.split('_')[1])], f"ShipmentCapacity_{u}_{v}")

        for count, group in enumerate(independent_set_inequalities):
            model.addCons(quicksum(vaccine_vars[node] for node in group) <= 1, name=f"clique_{count}")

        for count, group in enumerate(path_constraints):
            model.addCons(quicksum(vaccine_vars[node] for node in group) <= 1, name=f"path_{count}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_production_zones': 78,
        'n_distribution_zones': 100,
        'min_transport_cost': 48,
        'max_transport_cost': 600,
        'min_zone_cost': 22,
        'max_zone_cost': 70,
        'min_production_capacity': 307,
        'max_production_capacity': 1500,
        'min_budget_limit': 101,
        'max_budget_limit': 160,
        'max_transport_distance': 708,
        'graph_type': 'watts_strogatz',
        'edge_probability': 0.45,
        'affinity': 4,
        'ks': 12,
        'beta': 0.24,
    }

    optimizer = ExtendedVaccineDistributionOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")