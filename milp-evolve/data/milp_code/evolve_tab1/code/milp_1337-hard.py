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

class GarbageCollection:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.Number_of_Trucks > 0 and self.Number_of_Areas > 0
        assert self.Min_Truck_Cost >= 0 and self.Max_Truck_Cost >= self.Min_Truck_Cost
        assert self.Area_Cost_Lower_Bound >= 0 and self.Area_Cost_Upper_Bound >= self.Area_Cost_Lower_Bound

        truck_costs = np.random.randint(self.Min_Truck_Cost, self.Max_Truck_Cost + 1, self.Number_of_Trucks)
        area_costs = np.random.randint(self.Area_Cost_Lower_Bound, self.Area_Cost_Upper_Bound + 1, (self.Number_of_Trucks, self.Number_of_Areas))
        area_volumes = np.random.randint(1, 10, self.Number_of_Areas)
        
        emission_reductions = np.random.uniform(self.Min_Emission_Reduction, self.Max_Emission_Reduction, self.Number_of_Trucks)
        
        graph = Graph.barabasi_albert(self.Number_of_Trucks, self.Affinity)
        cliques = graph.efficient_greedy_clique_partition()
        incompatibilities = set(graph.edges)
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                incompatibilities.remove(edge)
            if len(clique) > 1:
                incompatibilities.add(clique)

        used_nodes = set()
        for group in incompatibilities:
            used_nodes.update(group)
        for node in range(self.Number_of_Trucks):
            if node not in used_nodes:
                incompatibilities.add((node,))
        
        return {
            "truck_costs": truck_costs,
            "area_costs": area_costs,
            "area_volumes": area_volumes,
            "emission_reductions": emission_reductions,
            "graph": graph,
            "incompatibilities": incompatibilities,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        truck_costs = instance['truck_costs']
        area_costs = instance['area_costs']
        area_volumes = instance['area_volumes']
        emission_reductions = instance['emission_reductions']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        
        model = Model("GarbageCollection")
        number_of_trucks = len(truck_costs)
        number_of_areas = len(area_costs[0])

        M = 10e6  # Big M for constraint relaxations
        
        # Decision variables
        truck_vars = {v: model.addVar(vtype="B", name=f"Truck_{v}") for v in range(number_of_trucks)}
        area_vars = {(v, r): model.addVar(vtype="B", name=f"Truck_{v}_Area_{r}") for v in range(number_of_trucks) for r in range(number_of_areas)}
        emission_vars = {v: model.addVar(vtype="C", name=f"Emission_{v}", lb=0) for v in range(number_of_trucks)}
        
        # Objective: minimize the total cost including truck startup costs and area covering costs
        model.setObjective(
            quicksum(truck_costs[v] * truck_vars[v] for v in range(number_of_trucks)) +
            quicksum(area_costs[v, r] * area_vars[v, r] for v in range(number_of_trucks) for r in range(number_of_areas)), "minimize"
        )
        
        # Set Covering Constraints: Each area must be covered by at least one truck
        for r in range(number_of_areas):
            model.addCons(quicksum(area_vars[v, r] for v in range(number_of_trucks)) >= 1, f"Area_{r}_Coverage")
        
        # Constraints: Only active trucks can serve areas
        for v in range(number_of_trucks):
            for r in range(number_of_areas):
                model.addCons(area_vars[v, r] <= truck_vars[v], f"Truck_{v}_Serve_{r}")
        
        # Constraints: Emission reduction targets
        for v in range(number_of_trucks):
            model.addCons(emission_vars[v] >= emission_reductions[v], f"Truck_{v}_Emission_Target")

        # Graph Incompatibilities Constraints
        for count, group in enumerate(incompatibilities):
            if isinstance(group, tuple):
                model.addCons(quicksum(truck_vars[node] for node in group) <= len(group) - 1, f"Incompatibility_{count}")
            else:
                model.addCons(truck_vars[group] <= 1, f"Incompatibility_{count}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    parameters = {
        'Number_of_Trucks': 50,
        'Number_of_Areas': 120,
        'Area_Cost_Lower_Bound': 3000,
        'Area_Cost_Upper_Bound': 5000,
        'Min_Truck_Cost': 3000,
        'Max_Truck_Cost': 10000,
        'Min_Emission_Reduction': 2625,
        'Max_Emission_Reduction': 1875,
        'Affinity': 18,
    }
    seed = 42

    garbage_collection_optimizer = GarbageCollection(parameters, seed=seed)
    instance = garbage_collection_optimizer.generate_instance()
    solve_status, solve_time, objective_value = garbage_collection_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")