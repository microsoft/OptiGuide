import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

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


class FactoryLocationDecision:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self):
        return Graph.erdos_renyi(self.n_nodes, self.edge_probability)

    def generate_instance(self):
        graph = self.generate_graph()
        demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        capacities = np.random.randint(100, 500, size=graph.number_of_nodes)
        setup_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        transportation_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        maintenance_costs = np.random.randint(20, 100, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'setup_costs': setup_costs,
            'transportation_costs': transportation_costs,
            'maintenance_costs': maintenance_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        setup_costs = instance['setup_costs']
        transportation_costs = instance['transportation_costs']
        maintenance_costs = instance['maintenance_costs']

        model = Model("FactoryLocationDecision")

        # Add variables
        factory_vars = {node: model.addVar(vtype="B", name=f"Factory_{node}") for node in graph.nodes}
        supply_vars = {(i, j): model.addVar(vtype="C", name=f"Supply_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Capacity Constraints for factories
        for factory in graph.nodes:
            model.addCons(quicksum(supply_vars[node, factory] for node in graph.nodes) <= capacities[factory] * factory_vars[factory], name=f"Capacity_{factory}")

        # Demand Constraints for neighborhoods
        for node in graph.nodes:
            model.addCons(quicksum(supply_vars[node, factory] for factory in graph.nodes) == demands[node], name=f"Demand_{node}")

        # Annual Maintenance Allocation
        for factory in graph.nodes:
            model.addCons(factory_vars[factory] * maintenance_costs[factory] <= maintenance_costs[factory], name=f"Maintenance_{factory}")

        # Objective: Minimize total cost
        setup_cost = quicksum(factory_vars[node] * setup_costs[node] for node in graph.nodes)
        trans_cost = quicksum(supply_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes)
        maintain_cost = quicksum(factory_vars[node] * maintenance_costs[node] for node in graph.nodes)

        total_cost = setup_cost + trans_cost + maintain_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 150,
        'edge_probability': 0.51,
        'graph_type': 'erdos_renyi',
    }

    factory_location_decision = FactoryLocationDecision(parameters, seed=seed)
    instance = factory_location_decision.generate_instance()
    solve_status, solve_time = factory_location_decision.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")