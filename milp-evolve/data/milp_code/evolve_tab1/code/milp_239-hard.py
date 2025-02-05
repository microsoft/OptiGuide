import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
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
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

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
############# Helper function #############

class MaximumHarvestedYield:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.number_of_crops, self.crop_connect_prob)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.number_of_crops, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()
        crop_exists_prob = np.random.uniform(0.8, 1, self.number_of_crops)
        crop_harvest_costs = np.random.randint(1, self.max_harvest_cost, self.number_of_crops)
        budget = np.random.randint(self.min_budget, self.max_budget)
        cliques = graph.efficient_greedy_clique_partition()
        constraints = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                constraints.remove(edge)
            if len(clique) > 1:
                constraints.add(clique)
        mature_crop_reward = np.random.randint(1, 100, self.number_of_crops)
        res = {'graph': graph,
               'constraints': constraints,
               'crop_exists_prob': crop_exists_prob,
               'crop_harvest_costs': crop_harvest_costs,
               'budget': budget,
               'mature_crop_reward': mature_crop_reward}
        res['crop_groups'] = [clique for clique in cliques if len(clique) <= self.max_clique_size]
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        constraints = instance['constraints']
        crop_exists_prob = instance['crop_exists_prob']
        crop_harvest_costs = instance['crop_harvest_costs']
        budget = instance['budget']
        crop_groups = instance['crop_groups']
        mature_crop_reward = instance['mature_crop_reward']

        model = Model("MaximumHarvestedYield")
        crop_vars = {node: model.addVar(vtype="B", name=f"x_{node}") for node in graph.nodes}

        for count, group in enumerate(constraints):
            if len(group) > 1:
                model.addCons(quicksum(crop_vars[node] for node in group) <= 1, name=f"CropConnectivityConstraint_{count}")

        crop_cost_constraints = quicksum(crop_harvest_costs[node] * crop_vars[node] for node in graph.nodes)
        model.addCons(crop_cost_constraints <= budget, name="NodeHarvestCost")

        objective_expr = quicksum((crop_exists_prob[node] + mature_crop_reward[node]) * crop_vars[node] for node in graph.nodes)

        for count, clique in enumerate(crop_groups):
            if len(clique) > 1:
                model.addCons(quicksum(crop_vars[node] for node in clique) <= 1, name=f"NeighboringNodeConstraint_{count}")

        for edge in graph.edges:
            u, v = edge
            model.addCons(crop_vars[u] + crop_vars[v] <= 1, name=f"ConnectivityConstraint_{u}_{v}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_crops': 842,
        'crop_connect_prob': 0.1,
        'affinity': 60,
        'graph_type': 'barabasi_albert',
        'max_harvest_cost': 2025,
        'min_budget': 10000,
        'max_budget': 15000,
        'max_clique_size': 687,
        'min_n': 370,
        'max_n': 1218,
        'er_prob': 0.17,
    }

    maximum_harvested_yield = MaximumHarvestedYield(parameters, seed=seed)
    instance = maximum_harvested_yield.generate_instance()
    solve_status, solve_time = maximum_harvested_yield.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")