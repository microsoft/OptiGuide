import random
import time
import numpy as np
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
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        weights = {edge: random.randint(self.weight_low, self.weight_high) for edge in graph.edges}
        land_costs = np.random.randint(10, 100, self.n_nodes)
        energy_availability = np.random.randint(50, 150, self.n_nodes)
        
        predicted_benefits = np.random.randint(1, 10, self.n_nodes)
        setup_costs = np.random.randint(100, 500, self.n_nodes)
        
        res = {'graph': graph, 'weights': weights, 'land_costs': land_costs,
               'energy_availability': energy_availability, 'predicted_benefits': predicted_benefits,
               'setup_costs': setup_costs}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        weights = instance['weights']
        land_costs = instance['land_costs']
        energy_availability = instance['energy_availability']
        predicted_benefits = instance['predicted_benefits']
        setup_costs = instance['setup_costs']

        model = Model("EVStationPlacement")
        x, y, s = {}, {}, {}

        for u in graph.nodes:
            x[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="x_%s" % u)
            s[u] = model.addVar(vtype='C', lb=0.0, ub=self.max_oper_capacity, name="s_%s" % u)
            model.addCons(s[u] >= self.min_oper_capacity * x[u], "MinOperCap_%s" % u)

        for e in graph.edges:
            y[e] = model.addVar(vtype='B', lb=0.0, ub=1, name="y_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= x[e[0]] + x[e[1]], "C1_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= 2 - x[e[0]] - x[e[1]], "C2_%s_%s" % (e[0], e[1]))

        for u in graph.nodes:
            model.addCons(x[u] * self.energy_demand <= energy_availability[u], "Energy_Lim_%s" % u)
            model.addCons(x[u] * setup_costs[u] <= self.budget, "Budget_Lim_%s" % u)

        # New Objective: Maximize accessibility + predicted benefits
        objective_expr = quicksum(weights[e] * y[e] for e in graph.edges) + \
                         quicksum(predicted_benefits[u] * x[u] for u in graph.nodes)

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_nodes': 450,
        'edge_probability': 0.17,
        'affinity': 2,
        'graph_type': 'barabasi_albert',
        'weight_low': 0,
        'weight_high': 250,
        'energy_demand': 37,
        'min_oper_capacity': 5,
        'max_oper_capacity': 75,
        'budget': 20000,
    }

    ev_station_placement = EVStationPlacement(parameters, seed=seed)
    instance = ev_station_placement.generate_instance()
    solve_status, solve_time = ev_station_placement.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")