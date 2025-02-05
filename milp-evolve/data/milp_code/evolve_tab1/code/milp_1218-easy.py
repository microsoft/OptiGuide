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

class HealthcareDistributionPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_centers > 0 and self.n_regions > 0
        assert self.min_center_cost >= 0 and self.max_center_cost >= self.min_center_cost
        assert self.min_region_cost >= 0 and self.max_region_cost >= self.min_region_cost
        assert self.min_center_capacity > 0 and self.max_center_capacity >= self.min_center_capacity

        center_costs = np.random.randint(self.min_center_cost, self.max_center_cost + 1, self.n_centers)
        region_costs = np.random.randint(self.min_region_cost, self.max_region_cost + 1, (self.n_centers, self.n_regions))
        capacities = np.random.randint(self.min_center_capacity, self.max_center_capacity + 1, self.n_centers)
        demands = np.random.randint(1, 10, self.n_regions)

        graph = Graph.barabasi_albert(self.n_centers, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_travel_times = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(self.n_centers):
            if node not in used_nodes:
                inequalities.add((node,))
        
        return {
            "center_costs": center_costs,
            "region_costs": region_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_travel_times": edge_travel_times,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        center_costs = instance['center_costs']
        region_costs = instance['region_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_travel_times = instance['edge_travel_times']
        
        model = Model("HealthcareDistributionPlanning")
        n_centers = len(center_costs)
        n_regions = len(region_costs[0])
        
        # Decision variables
        center_vars = {c: model.addVar(vtype="B", name=f"Center_{c}") for c in range(n_centers)}
        region_vars = {(c, r): model.addVar(vtype="B", name=f"Center_{c}_Region_{r}") for c in range(n_centers) for r in range(n_regions)}
        connection_vars = {edge: model.addVar(vtype="B", name=f"Connection_{edge[0]}_{edge[1]}") for edge in graph.edges}
        
        # Objective: minimize the total cost including center costs, region servicing costs, and travel times
        model.setObjective(
            quicksum(center_costs[c] * center_vars[c] for c in range(n_centers)) +
            quicksum(region_costs[c, r] * region_vars[c, r] for c in range(n_centers) for r in range(n_regions)) +
            quicksum(edge_travel_times[i] * connection_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )
        
        # Constraints: Each region is covered by at least one center
        for r in range(n_regions):
            model.addCons(quicksum(region_vars[c, r] for c in range(n_centers)) >= 1, f"Region_{r}_Service")
        
        # Constraints: Only open centers can cover regions
        for c in range(n_centers):
            for r in range(n_regions):
                model.addCons(region_vars[c, r] <= center_vars[c], f"Center_{c}_Serve_{r}")
        
        # Constraints: Centers cannot exceed their capacity
        for c in range(n_centers):
            model.addCons(quicksum(demands[r] * region_vars[c, r] for r in range(n_regions)) <= capacities[c], f"Center_{c}_Capacity")
        
        # Constraints: Center Graph Cliques
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(center_vars[node] for node in group) <= 1, f"Clique_{count}")
        
        # Adding complexity: additional clique constraints
        complexities = list(inequalities)  # convert set to list to manipulate
        
        # Additional larger cliques formed from subsets of existing cliques
        # Let's assume the complexity involves adding cliques of size 3 if possible
        new_cliques = []
        for clique in complexities:
            if isinstance(clique, tuple) and len(clique) >= 3:
                new_cliques.extend(list(combinations(clique, 3)))
        
        complexity_vars = {idx: model.addVar(vtype="B", name=f"Complexity_{idx}") for idx in range(len(new_cliques))}
        
        for idx, clique in enumerate(new_cliques):
            model.addCons(quicksum(center_vars[node] for node in clique) <= complexity_vars[idx], f"Complexity_Clique_{idx}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 37,
        'n_regions': 112,
        'min_region_cost': 343,
        'max_region_cost': 3000,
        'min_center_cost': 1686,
        'max_center_cost': 5000,
        'min_center_capacity': 555,
        'max_center_capacity': 1260,
        'affinity': 11,
    }

    healthcare_optimizer = HealthcareDistributionPlanning(parameters, seed)
    instance = healthcare_optimizer.generate_instance()
    solve_status, solve_time, objective_value = healthcare_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")