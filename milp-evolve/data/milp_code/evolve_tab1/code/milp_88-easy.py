import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
class HyperGraph:
    """
    Helper function: Container for a hypergraph.
    """
    def __init__(self, number_of_nodes, hyperedges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.hyperedges = hyperedges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def random_hypergraph(number_of_nodes, number_of_hyperedges, hyperedge_size):
        """
        Generate a random hypergraph.
        """
        hyperedges = []
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        
        for _ in range(number_of_hyperedges):
            hyperedge = tuple(np.random.choice(number_of_nodes, hyperedge_size, replace=False))
            hyperedges.append(hyperedge)
            for node in hyperedge:
                degrees[node] += 1
                neighbors[node].update(hyperedge)
        
        return HyperGraph(number_of_nodes, hyperedges, degrees, neighbors)

############# Helper function #############

class NetworkBandwidthOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_hypergraph(self):
        hypergraph = HyperGraph.random_hypergraph(self.n_nodes, self.n_hyperedges, self.hyperedge_size)
        return hypergraph

    def generate_instance(self):
        hypergraph = self.generate_hypergraph()
        
        # Generate node capacities and hyperedge bandwidths
        node_capacities = np.random.randint(10, 100, self.n_nodes)
        hyperedge_bandwidths = np.random.uniform(1, 10, self.n_hyperedges)
        
        connection_requirement = np.random.randint(5, 20, self.n_hyperedges)
        
        res = {
            'hypergraph': hypergraph,
            'node_capacities': node_capacities,
            'hyperedge_bandwidths': hyperedge_bandwidths,
            'connection_requirement': connection_requirement
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hypergraph = instance['hypergraph']
        node_capacities = instance['node_capacities']
        hyperedge_bandwidths = instance['hyperedge_bandwidths']
        connection_requirement = instance['connection_requirement']

        model = Model("NetworkBandwidthOptimization")
        
        node_vars = {node: model.addVar(vtype="B", name=f"node_{node}") for node in hypergraph.nodes}
        hyperedge_vars = {i: model.addVar(vtype="B", name=f"hyperedge_{i}") for i in range(len(hypergraph.hyperedges))}
        
        for i, hyperedge in enumerate(hypergraph.hyperedges):
            for node in hyperedge:
                model.addCons(node_vars[node] >= hyperedge_vars[i], name=f"connection_{i}_{node}")

        for node in hypergraph.nodes:
            model.addCons(quicksum(hyperedge_bandwidths[i] * hyperedge_vars[i] 
                                   for i, hyperedge in enumerate(hypergraph.hyperedges) if node in hyperedge) <= 
                          node_capacities[node], name=f"capacity_{node}")
        
        objective_expr = quicksum(connection_requirement[i] * hyperedge_vars[i] for i in range(len(hypergraph.hyperedges)))
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 800,
        'n_hyperedges': 350,
        'hyperedge_size': 21,
    }

    network_bandwidth_problem = NetworkBandwidthOptimization(parameters, seed=seed)
    instance = network_bandwidth_problem.generate_instance()
    solve_status, solve_time = network_bandwidth_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")