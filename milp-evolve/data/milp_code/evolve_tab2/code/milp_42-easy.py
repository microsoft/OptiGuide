import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
class Graph:
    """
    Helper function: Container for a graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.
        """
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
        """
        Generate a Barabási-Albert random graph with a given edge probability.
        """
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

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph
############# Helper function #############

class EnhancedGraphProblem:
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
            graph = Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity) 
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()
        weights = {edge: random.randint(self.weight_low, self.weight_high) for edge in graph.edges}
        
        # Generate data related to land cost, zoning regulations, and energy limits
        land_costs = np.random.randint(10, 100, self.n_nodes)
        zoning_compatibility = np.random.randint(0, 2, self.n_nodes)
        energy_availability = np.random.randint(50, 150, self.n_nodes)
        grid_substations = np.random.randint(0, self.n_nodes // 3, self.n_nodes)
        
        # New data: Tariff impacts and deployment order constraints
        tariff_impacts = np.random.normal(self.tariff_mean, self.tariff_std, size=self.n_nodes)
        deployment_order = sorted(random.sample(range(self.n_nodes), self.deployment_phases))

        res = {'graph': graph, 'weights': weights, 'land_costs': land_costs,
               'zoning_compatibility': zoning_compatibility, 'energy_availability': energy_availability,
               'grid_substations': grid_substations, 'tariff_impacts': tariff_impacts,
               'deployment_order': deployment_order}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        weights = instance['weights']
        land_costs = instance['land_costs']
        zoning_compatibility = instance['zoning_compatibility']
        energy_availability = instance['energy_availability']
        grid_substations = instance['grid_substations']
        tariff_impacts = instance['tariff_impacts']
        deployment_order = instance['deployment_order']

        model = Model("EnhancedGraphProblem")
        max_colors = self.max_colors
        var_names = {}
        color_used = [model.addVar(vtype="B", name=f"color_used_{color}") for color in range(max_colors)]

        # Variables: x[i, c] = 1 if node i is colored with color c, 0 otherwise
        for node in graph.nodes:
            for color in range(max_colors):
                var_names[(node, color)] = model.addVar(vtype="B", name=f"x_{node}_{color}")
                model.addCons(var_names[(node, color)] <= color_used[color])

        # New Variables: Node is used for placement or in use by graph properties
        use_node = {node: model.addVar(vtype='B', name=f"use_node_{node}") for node in graph.nodes}
        tariff_impact_node = {node: model.addVar(vtype='C', name=f"tariff_impact_{node}") for node in graph.nodes}
        start_time = {node: model.addVar(vtype='C', name=f"start_time_{node}") for node in graph.nodes}

        # Constraint: Each node must have exactly one color
        for node in graph.nodes:
            model.addCons(quicksum(var_names[(node, color)] for color in range(max_colors)) == 1)

        # Constraint: Adjacent nodes must have different colors
        for edge in graph.edges:
            node_u, node_v = edge
            for color in range(max_colors):
                model.addCons(var_names[(node_u, color)] + var_names[(node_v, color)] <= 1)

        # Objective: Enhance by incorporating the penalty for land use and constraints like zoning
        energy_demand = self.energy_demand
        color_penalty = self.color_penalty

        for node in graph.nodes:
            model.addCons(var_names[(node, 0)] * energy_demand <= energy_availability[node])
            model.addCons(use_node[node] <= zoning_compatibility[node])
            # Tariff impact constraint
            model.addCons(tariff_impact_node[node] == tariff_impacts[node] * use_node[node])

        # Sequential deployment constraints
        for i in range(len(deployment_order) - 1):
            u = deployment_order[i]
            v = deployment_order[i+1]
            model.addCons(start_time[v] >= start_time[u] + use_node[u])

        objective_expr = quicksum(color_used[color] * color_penalty for color in range(max_colors)) \
                         - quicksum(land_costs[node] * use_node[node] for node in graph.nodes) \
                         + quicksum(grid_substations[node] * use_node[node] for node in graph.nodes) \
                         - quicksum(tariff_impact_node[node] for node in graph.nodes)
        
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_nodes': 500,
        'edge_probability': 0.52,
        'affinity': 4,
        'graph_type': 'barabasi_albert',
        'max_colors': 10,
        'color_penalty': 1.0,
        'weight_low': 1,
        'weight_high': 100,
        'energy_demand': 75,
        'tariff_mean': 3.0,
        'tariff_std': 0.5,
        'deployment_phases': 5,
    }

    enhanced_graph_problem = EnhancedGraphProblem(parameters, seed=seed)
    instance = enhanced_graph_problem.generate_instance()
    solve_status, solve_time = enhanced_graph_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")