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

    def efficient_greedy_clique_partition(self):
        """
        Partition the graph into cliques using an efficient greedy algorithm.
        """
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

class MarineProtectedAreas:
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
        
        # Generate ecological activity probabilities
        MarineExploits = np.random.uniform(0.8, 1, self.n_nodes)
        
        # Generate pollution control activities and piecewise linear pollution reduction costs with deviations
        Nearshore = {edge: [np.random.uniform(1, 5), np.random.uniform(5, 10)] for edge in graph.edges if np.random.random() < self.nearshore_prob}
        Nearshore_deviation = {edge: np.random.uniform(0, 1) for edge in Nearshore}
        
        # Generate node pollution concentrations with deviations
        NodeResidues = np.random.randint(1, self.max_residue, self.n_nodes)
        NodeResidues_deviation = np.random.uniform(0, 1, self.n_nodes)
        HazardThreshold = np.random.randint(self.min_threshold, self.max_threshold)
        
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)
        
        res = {'graph': graph,
               'inequalities': inequalities,
               'MarineExploits': MarineExploits,
               'Nearshore': Nearshore,
               'Nearshore_deviation': Nearshore_deviation,
               'NodeResidues': NodeResidues,
               'NodeResidues_deviation': NodeResidues_deviation,
               'HazardThreshold': HazardThreshold}
        
        ### new instance data for pollution control ###
        res['SeabedSurveillance'] = [clique for clique in cliques if len(clique) <= self.max_clique_size]
        
        ### given instance data code ends here
        ### new instance data code ends here
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        MarineExploits = instance['MarineExploits']
        Nearshore = instance['Nearshore']
        Nearshore_deviation = instance['Nearshore_deviation']
        NodeResidues = instance['NodeResidues']
        NodeResidues_deviation = instance['NodeResidues_deviation']
        HazardThreshold = instance['HazardThreshold']
        SeabedSurveillance = instance['SeabedSurveillance']

        model = Model("MarineProtectedAreas")
        var_names = {}
        edge_vars = {}
        plf_vars = {}

        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        # Add edge variables for linear conversion
        for edge, costs in Nearshore.items():
            u, v = edge
            edge_vars[edge] = model.addVar(vtype="B", name=f"y_{u}_{v}")
            plf_vars[edge] = [model.addVar(vtype="B", name=f"plf_{u}_{v}_1"), model.addVar(vtype="B", name=f"plf_{u}_{v}_2")]
            model.addCons(plf_vars[edge][0] + plf_vars[edge][1] <= 1, name=f"plf_{u}_{v}_sum")
            model.addCons(edge_vars[edge] <= plf_vars[edge][0] + plf_vars[edge][1], name=f"plf_{u}_{v}_link")

        for count, group in enumerate(inequalities):
            if len(group) > 1:
                model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        for u, v in Nearshore:
            model.addCons(var_names[u] + var_names[v] - edge_vars[(u, v)] <= 1, name=f"edge_{u}_{v}")

        # Define the robust pollution control constraint
        pollution_constraints = quicksum((NodeResidues[node] + NodeResidues_deviation[node]) * var_names[node] for node in graph.nodes)
        model.addCons(pollution_constraints <= HazardThreshold, name="contaminant_assessment")

        # Define objective to include pollution control costs
        objective_expr = quicksum(MarineExploits[node] * var_names[node] for node in graph.nodes)
        objective_expr -= quicksum(Nearshore[edge][0] * plf_vars[edge][0] + Nearshore[edge][1] * plf_vars[edge][1] for edge in Nearshore)
        
        for count, clique in enumerate(SeabedSurveillance):
            if len(clique) > 1:
                model.addCons(quicksum(var_names[node] for node in clique) <= 1, name=f"enhanced_clique_{count}")

        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 421,
        'edge_probability': 0.73,
        'affinity': 36,
        'graph_type': 'barabasi_albert',
        'nearshore_prob': 0.1,
        'max_residue': 2700,
        'min_threshold': 10000,
        'max_threshold': 15000,
        'max_clique_size': 1225,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    marine_protected_areas = MarineProtectedAreas(parameters, seed=seed)
    instance = marine_protected_areas.generate_instance()
    solve_status, solve_time = marine_protected_areas.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")