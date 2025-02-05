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

class IndependentSet:
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

        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        res = {'graph': graph,
               'inequalities': inequalities}

        # New instance data for constraints and objectives
        number_of_vehicles = 10
        vehicle_capacities = np.random.randint(1, 10, number_of_vehicles)
        supply_urgency_levels = np.random.randint(1, 5, len(graph.nodes))

        res.update({
            'number_of_vehicles': number_of_vehicles,
            'vehicle_capacities': vehicle_capacities,
            'supply_urgency_levels': supply_urgency_levels
        })

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        number_of_vehicles = instance['number_of_vehicles']
        vehicle_capacities = instance['vehicle_capacities']
        supply_urgency_levels = instance['supply_urgency_levels']

        model = Model("AllocationOfMedicalSupplies")
        var_names = {}
        vehicle_vars = {}
        transport_vars = {}

        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        for v in range(number_of_vehicles):
            vehicle_vars[v] = model.addVar(vtype="B", name=f"vehicle_{v}")

        for node in graph.nodes:
            for v in range(number_of_vehicles):
                transport_vars[(node, v)] = model.addVar(vtype="B", name=f"transport_{node}_{v}")

        for count, group in enumerate(inequalities):
            if len(group) > 1:
                model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        for node in graph.nodes:
            model.addCons(quicksum(transport_vars[(node, v)] for v in range(number_of_vehicles)) == var_names[node],
                          name=f"supply_transport_{node}")

        for v in range(number_of_vehicles):
            model.addCons(quicksum(transport_vars[(node, v)] * supply_urgency_levels[node] for node in graph.nodes) <= vehicle_capacities[v],
                          name=f"vehicle_capacity_{v}")

        for node in graph.nodes:
            model.addCons(var_names[node] * supply_urgency_levels[node] <= 1, name=f"urgency_{node}")

        # Objective: Balance maximization of independent distribution centers and minimization of utilized distribution centers
        objective_expr = quicksum(var_names[node] for node in graph.nodes) - quicksum(vehicle_vars[v] for v in range(number_of_vehicles))
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 750,
        'edge_probability': 0.3,
        'affinity': 32,
        'graph_type': 'barabasi_albert',
    }

    independent_set_problem = IndependentSet(parameters, seed=seed)
    instance = independent_set_problem.generate_instance()
    solve_status, solve_time = independent_set_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")