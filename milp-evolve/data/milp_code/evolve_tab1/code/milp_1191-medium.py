import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

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

class UrbanSchoolPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_schools > 0 and self.n_neighborhoods > 0
        assert self.min_school_cost >= 0 and self.max_school_cost >= self.min_school_cost
        assert self.min_neighborhood_cost >= 0 and self.max_neighborhood_cost >= self.min_neighborhood_cost
        assert self.min_school_capacity > 0 and self.max_school_capacity >= self.min_school_capacity

        school_costs = np.random.randint(self.min_school_cost, self.max_school_cost + 1, self.n_schools)
        neighborhood_costs = np.random.randint(self.min_neighborhood_cost, self.max_neighborhood_cost + 1, (self.n_schools, self.n_neighborhoods))
        capacities = np.random.randint(self.min_school_capacity, self.max_school_capacity + 1, self.n_schools)
        demands = np.random.randint(10, 100, self.n_neighborhoods)

        graph = Graph.barabasi_albert(self.n_schools, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(self.n_schools):
            if node not in used_nodes:
                inequalities.add((node,))
        
        return {
            "school_costs": school_costs,
            "neighborhood_costs": neighborhood_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_weights": edge_weights,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        school_costs = instance['school_costs']
        neighborhood_costs = instance['neighborhood_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_weights = instance['edge_weights']
        
        model = Model("UrbanSchoolPlanning")
        n_schools = len(school_costs)
        n_neighborhoods = len(neighborhood_costs[0])
        
        # Decision variables
        school_vars = {s: model.addVar(vtype="B", name=f"School_{s}") for s in range(n_schools)}
        neighborhood_vars = {(s, n): model.addVar(vtype="B", name=f"School_{s}_Neighborhood_{n}") for s in range(n_schools) for n in range(n_neighborhoods)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        
        # Objective: minimize the total cost including school costs and neighborhood costs
        model.setObjective(
            quicksum(school_costs[s] * school_vars[s] for s in range(n_schools)) +
            quicksum(neighborhood_costs[s, n] * neighborhood_vars[s, n] for s in range(n_schools) for n in range(n_neighborhoods)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )
        
        # Constraints: Each neighborhood demand is met by exactly one school
        for n in range(n_neighborhoods):
            model.addCons(quicksum(neighborhood_vars[s, n] for s in range(n_schools)) == 1, f"Neighborhood_{n}_Demand")
        
        # Constraints: Only open schools can serve neighborhoods
        for s in range(n_schools):
            for n in range(n_neighborhoods):
                model.addCons(neighborhood_vars[s, n] <= school_vars[s], f"School_{s}_Serve_{n}")
        
        # Constraints: Schools cannot exceed their capacity
        for s in range(n_schools):
            model.addCons(quicksum(demands[n] * neighborhood_vars[s, n] for n in range(n_neighborhoods)) <= capacities[s], f"School_{s}_Capacity")
        
        M = max(school_costs) + 1  # Big M constant for the formulation
        # Constraints: School Graph Cliques with Big M
        for count, group in enumerate(inequalities):
            if len(group) > 1:
                for u, v in combinations(group, 2):
                    model.addCons(school_vars[u] + school_vars[v] <= edge_vars[(u, v)] + 1, f"CliqueBigM_{u}_{v}")
                    model.addCons(edge_vars[(u, v)] <= school_vars[u], f"EdgeBigM_u_{u}_{v}")
                    model.addCons(edge_vars[(u, v)] <= school_vars[v], f"EdgeBigM_u_{v}_{u}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_schools': 55,
        'n_neighborhoods': 112,
        'min_neighborhood_cost': 6,
        'max_neighborhood_cost': 3000,
        'min_school_cost': 1264,
        'max_school_cost': 5000,
        'min_school_capacity': 46,
        'max_school_capacity': 1416,
        'affinity': 44,
    }

    school_planning_optimizer = UrbanSchoolPlanning(parameters, seed=42)
    instance = school_planning_optimizer.generate_instance()
    solve_status, solve_time, objective_value = school_planning_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")