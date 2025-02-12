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

class ComplexGraphColoring:
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
    
    def generate_resource_requirements(self, cliques):
        return {tuple(clique): random.randint(1, self.max_resource) for clique in cliques}

    def generate_instance(self):
        graph = self.generate_graph()
        cliques = self.find_maximal_cliques(graph)
        resource_requirements = self.generate_resource_requirements(cliques)

        res = {'graph': graph, 'cliques': cliques, 'resource_requirements': resource_requirements}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        cliques = instance['cliques']
        resource_requirements = instance['resource_requirements']
        model = Model("ComplexGraphColoring")
        max_colors = self.max_colors
        var_names = {}

        # Variables: x[i, c] = 1 if node i is colored with color c, 0 otherwise
        for node in graph.nodes:
            for color in range(max_colors):
                var_names[(node, color)] = model.addVar(vtype="B", name=f"x_{node}_{color}")

        # Constraint: each node must have exactly one color
        for node in graph.nodes:
            model.addCons(quicksum(var_names[(node, color)] for color in range(max_colors)) == 1)

        # Clique and Resource inequalities
        resource_used = model.addVar(vtype="C", name="resource_used")
        for clique in cliques:
            if len(clique) > max_colors:
                raise ValueError("Not enough colors to satisfy clique constraints")
            for color in range(max_colors):
                model.addCons(quicksum(var_names[(node, color)] for node in clique) <= 1)
            # Resource constraint on clique coloring
            model.addCons(quicksum(var_names[(node, color)] for node in clique for color in range(max_colors)) <= resource_used * resource_requirements[tuple(clique)])

        # Objective: minimize the total cost, now consider resource usage cost
        color_used = [model.addVar(vtype="B", name=f"color_used_{color}") for color in range(max_colors)]
        for node in graph.nodes:
            for color in range(max_colors):
                model.addCons(var_names[(node, color)] <= color_used[color])
        
        objective_expr = quicksum(color_used) + resource_used
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_nodes': 1500,
        'edge_probability': 0.45,
        'affinity': 2,
        'graph_type': 'barabasi_albert',
        'max_colors': 7,
        'max_resource': 15,
    }

    complex_graph_coloring_problem = ComplexGraphColoring(parameters, seed=seed)
    instance = complex_graph_coloring_problem.generate_instance()
    solve_status, solve_time = complex_graph_coloring_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")