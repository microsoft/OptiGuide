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

class LoyaltyRewards:
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
        P = np.random.randint(5, 30, size=(self.n_regions, self.n_members))
        delta = np.random.normal(0, 2, size=(self.n_regions, self.n_members))
        region_limits = np.random.randint(50, 100, size=self.n_regions)
        benefits_weights = np.random.randint(1, 10, size=self.n_members)

        res = {'P': P, 'delta': delta, 'region_limits': region_limits, 'benefits_weights': benefits_weights}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        P = instance['P']
        delta = instance['delta']
        region_limits = instance['region_limits']
        benefits_weights = instance['benefits_weights']
        
        # build the optimization model
        model = Model("LoyaltyRewards")

        y = {}
        for j in range(self.n_members):
            y[j] = model.addVar(vtype='B', lb=0.0, ub=1, name="y_%s" % (j+1))
        
        for i in range(self.n_regions):
            model.addCons(quicksum((P[i, j] + delta[i, j]) * y[j] for j in range(self.n_members)) <= region_limits[i], "RegionLimit_Robust_%s" % (i+1))

        objective_expr = quicksum(benefits_weights[j] * y[j] for j in range(self.n_members))
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_regions': 300,
        'n_members': 900,
    }

    loyalty_rewards = LoyaltyRewards(parameters, seed=seed)
    instance = loyalty_rewards.generate_instance()
    solve_status, solve_time = loyalty_rewards.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")