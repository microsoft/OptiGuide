import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

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

class InfrastructureDevelopmentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_neighborhoods > 0 and self.n_zones > 0
        assert self.min_project_cost >= 0 and self.max_project_cost >= self.min_project_cost
        assert self.min_development_cost >= 0 and self.max_development_cost >= self.min_development_cost
        assert self.min_project_time > 0 and self.max_project_time >= self.min_project_time

        development_costs = np.random.randint(self.min_development_cost, self.max_development_cost + 1, self.n_neighborhoods)
        project_costs = np.random.randint(self.min_project_cost, self.max_project_cost + 1, (self.n_neighborhoods, self.n_zones))
        project_times = np.random.randint(self.min_project_time, self.max_project_time + 1, self.n_neighborhoods)
        zone_importance = np.random.randint(1, 10, self.n_zones)

        graph = Graph.barabasi_albert(self.n_neighborhoods, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        segment_costs = np.random.randint(1, 10, size=len(graph.edges))

        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(10):
            if node not in used_nodes:
                inequalities.add((node,))

        return {
            "development_costs": development_costs,
            "project_costs": project_costs,
            "project_times": project_times,
            "zone_importance": zone_importance,
            "graph": graph,
            "inequalities": inequalities,
            "segment_costs": segment_costs
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        development_costs = instance['development_costs']
        project_costs = instance['project_costs']
        project_times = instance['project_times']
        zone_importance = instance['zone_importance']
        graph = instance['graph']
        inequalities = instance['inequalities']
        segment_costs = instance['segment_costs']

        model = Model("InfrastructureDevelopmentOptimization")
        n_neighborhoods = len(development_costs)
        n_zones = len(project_costs[0])

        # Decision variables
        NeighborhoodExpansion_vars = {f: model.addVar(vtype="B", name=f"NeighborhoodExpansion_{f}") for f in range(n_neighborhoods)}
        ZoneDevelopment_vars = {(f, z): model.addVar(vtype="B", name=f"NeighborhoodExpansion_{f}_Zone_{z}") for f in range(n_neighborhoods) for z in range(n_zones)}
        HighCostProjects_vars = {edge: model.addVar(vtype="B", name=f"HighCostProjects_{edge[0]}_{edge[1]}") for edge in graph.edges}

        # New Variables for Big M Formulation
        MegaProjects_vars = {f: model.addVar(vtype="B", name=f"MegaProjects_{f}") for f in range(n_neighborhoods)}

        # Objective: minimize the total development cost and segment costs
        model.setObjective(
            quicksum(development_costs[f] * NeighborhoodExpansion_vars[f] for f in range(n_neighborhoods)) +
            quicksum(project_costs[f, z] * ZoneDevelopment_vars[f, z] for f in range(n_neighborhoods) for z in range(n_zones)) +
            quicksum(segment_costs[i] * HighCostProjects_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )

        # Constraints: Each zone must be developed by at least one project
        for z in range(n_zones):
            model.addCons(quicksum(ZoneDevelopment_vars[f, z] for f in range(n_neighborhoods)) >= 1, f"Zone_{z}_Development")

        # Constraints: Only selected expansions can develop zones
        for f in range(n_neighborhoods):
            for z in range(n_zones):
                model.addCons(ZoneDevelopment_vars[f, z] <= NeighborhoodExpansion_vars[f], f"NeighborhoodExpansion_{f}_Zone_{z}")

        # Constraints: Projects cannot exceed their time capacity
        for f in range(n_neighborhoods):
            model.addCons(quicksum(zone_importance[z] * ZoneDevelopment_vars[f, z] for z in range(n_zones)) <= project_times[f], f"NeighborhoodExpansion_{f}_Time")

        # Constraints: Graph Cliques for minimizing development inefficiencies
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(NeighborhoodExpansion_vars[node] for node in group) <= 1, f"Clique_{count}")

        # New Constraints: High-cost Projects Based on Big M Formulation
        M = self.M  # A large constant for Big M Formulation
        for edge in graph.edges:
            f1, f2 = edge
            model.addCons(MegaProjects_vars[f1] + MegaProjects_vars[f2] >= HighCostProjects_vars[edge], f"MegaProjectEdge_{f1}_{f2}")
            model.addCons(HighCostProjects_vars[edge] <= NeighborhoodExpansion_vars[f1] + NeighborhoodExpansion_vars[f2], f"EdgeEnforcement_{f1}_{f2}")
            model.addCons(MegaProjects_vars[f1] <= M * NeighborhoodExpansion_vars[f1], f"BigM_HighCost_{f1}")
            model.addCons(MegaProjects_vars[f2] <= M * NeighborhoodExpansion_vars[f2], f"BigM_HighCost_{f2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_neighborhoods': 30,
        'n_zones': 280,
        'min_project_cost': 0,
        'max_project_cost': 375,
        'min_development_cost': 281,
        'max_development_cost': 5000,
        'min_project_time': 200,
        'max_project_time': 1350,
        'affinity': 6,
        'M': 10000,
    }

    infra_optimizer = InfrastructureDevelopmentOptimization(parameters, seed)
    instance = infra_optimizer.generate_instance()
    solve_status, solve_time, objective_value = infra_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")