import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

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

class GraphColoring:
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

    def find_maximal_cliques(self, graph):
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(graph.edges)
        cliques = list(nx.find_cliques(nx_graph))
        return cliques

    def generate_instance(self):
        graph = self.generate_graph()
        cliques = self.find_maximal_cliques(graph)
        
        # Generate subgraphs for potential new constraints
        subgraphs = list(nx.connected_components(nx.Graph(graph.edges)))
        
        res = {'graph': graph, 'cliques': cliques, 'subgraphs': subgraphs}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        cliques = instance['cliques']
        subgraphs = instance['subgraphs']
        
        model = Model("GraphColoring")
        max_colors = self.max_colors
        var_names = {}

        # Variables: x[i, c] = 1 if node i is colored with color c, 0 otherwise
        for node in graph.nodes:
            for color in range(max_colors):
                var_names[(node, color)] = model.addVar(vtype="B", name=f"x_{node}_{color}")

        # Constraint: each node must have exactly one color
        for node in graph.nodes:
            model.addCons(quicksum(var_names[(node, color)] for color in range(max_colors)) == 1)

        # Clique inequalities: all nodes in a clique must have different colors
        for clique in cliques:
            if len(clique) > max_colors:
                raise ValueError("Not enough colors to satisfy clique constraints")
            for color in range(max_colors):
                model.addCons(quicksum(var_names[(node, color)] for node in clique) <= 1)

        # Objective: minimize the number of colors used
        color_used = [model.addVar(vtype="B", name=f"color_used_{color}") for color in range(max_colors)]
        for node in graph.nodes:
            for color in range(max_colors):
                model.addCons(var_names[(node, color)] <= color_used[color])
        
        # New Big M constraints for hierarchical color dependencies
        subgraph_dependency = {}
        for index, subgraph in enumerate(subgraphs):
            subgraph_dependency[index] = model.addVar(vtype="B", name=f"subgraph_dependency_{index}")
            for color in range(max_colors):
                # Introduce a constraint using Big M formulation
                M = len(graph.nodes)
                cond_expr = quicksum(var_names[(node, color)] for node in subgraph) - subgraph_dependency[index] * M
                model.addCons(cond_expr <= self.color_dependency_threshold)

        objective_expr = quicksum(color_used)
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_nodes': 1500,
        'edge_probability': 0.66,
        'affinity': 4,
        'graph_type': 'barabasi_albert',
        'max_colors': 5,
        'color_dependency_threshold': 2,
    }

    graph_coloring_problem = GraphColoring(parameters, seed=seed)
    instance = graph_coloring_problem.generate_instance()
    solve_status, solve_time = graph_coloring_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")