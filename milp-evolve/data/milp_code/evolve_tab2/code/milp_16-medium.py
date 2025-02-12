import random
import time
import numpy as np
from itertools import combinations, product
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

class EVStationPlacement:
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

        # Resource balance parameters
        grid_substations = np.random.randint(0, self.n_nodes // 3, self.n_nodes)
        
        res = {'graph': graph, 'weights': weights, 'land_costs': land_costs,
               'zoning_compatibility': zoning_compatibility, 'energy_availability': energy_availability,
               'grid_substations': grid_substations}

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        weights = instance['weights']
        land_costs = instance['land_costs']
        zoning_compatibility = instance['zoning_compatibility']
        energy_availability = instance['energy_availability']
        grid_substations = instance['grid_substations']

        model = Model("EVStationPlacement")

        x = {}  # Variable to indicate if a station is placed at a node
        y = {}  # Variable for max-cut problem
        z = {}  # Variable to indicate land use
        f = {}  # New: Reserve allocation variable to match protein folding

        for u in graph.nodes:
            x[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="x_%s" % u)
            z[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="z_%s" % u)
            f[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="f_%s" % u)
            
            # Land use cost constraints
            model.addCons(z[u] <= zoning_compatibility[u], "Zoning_Compat_%s" % u)

        for e in graph.edges:  # e=(u, v)
            y[e] = model.addVar(vtype='B', lb=0.0, ub=1, name="y_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= x[e[0]] + x[e[1]], "C1_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= 2 - x[e[0]] - x[e[1]], "C2_%s_%s" % (e[0], e[1]))

        # Energy availability constraints
        for u in graph.nodes:
            model.addCons(x[u] * self.energy_demand <= energy_availability[u], "Energy_Lim_%s" % u)
            # Matching resource availability with grid substation proxies
            model.addCons(x[u] + f[u] <= 1, "Resource_Match_%s" % u)

        # Objectives: maximize accessibility while minimizing land cost
        objective_expr = quicksum(weights[e] * y[e] for e in graph.edges) - \
                         quicksum(land_costs[u] * z[u] for u in graph.nodes) + \
                         quicksum(grid_substations[u] * f[u] for u in graph.nodes)

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_nodes': 150,
        'edge_probability': 0.3,
        'affinity': 8,
        'graph_type': 'barabasi_albert',
        'weight_low': 0,
        'weight_high': 50,
        'energy_demand': 75,
    }

    ev_station_placement = EVStationPlacement(parameters, seed=seed)
    instance = ev_station_placement.generate_instance()
    solve_status, solve_time = ev_station_placement.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")