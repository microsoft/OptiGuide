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

class HealthcareFacilityOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods > 0
        assert self.min_travel_cost >= 0 and self.max_travel_cost >= self.min_travel_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_hospital_capacity > 0 and self.max_hospital_capacity >= self.min_hospital_capacity
        
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        travel_costs = np.random.randint(self.min_travel_cost, self.max_travel_cost + 1, (self.n_facilities, self.n_neighborhoods))
        capacities = np.random.randint(self.min_hospital_capacity, self.max_hospital_capacity + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_neighborhoods)

        graph = Graph.barabasi_albert(self.n_facilities, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_dangers = np.random.randint(1, 10, size=len(graph.edges))
        
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
            "facility_costs": facility_costs,
            "travel_costs": travel_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_dangers": edge_dangers
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        travel_costs = instance['travel_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_dangers = instance['edge_dangers']
        
        model = Model("HealthcareFacilityOptimization")
        n_facilities = len(facility_costs)
        n_neighborhoods = len(travel_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        assign_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighbor_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        
        # Objective: minimize the total cost and danger levels
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(travel_costs[f, n] * assign_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) +
            quicksum(edge_dangers[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )
        
        # Constraints: Each neighborhood demand is met by exactly one facility
        for n in range(n_neighborhoods):
            model.addCons(quicksum(assign_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Demand")
        
        # Constraints: Only open facilities can serve neighborhoods
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(assign_vars[f, n] <= facility_vars[f], f"Facility_{f}_Serve_{n}")
        
        # Constraints: Facilities cannot exceed their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(demands[n] * assign_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")
        
        # Constraints: Facility Graph Cliques (to minimize danger levels)
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(facility_vars[node] for node in group) <= 1, f"Clique_{count}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 50,
        'n_neighborhoods': 224,
        'min_travel_cost': 5,
        'max_travel_cost': 1500,
        'min_facility_cost': 750,
        'max_facility_cost': 5000,
        'min_hospital_capacity': 675,
        'max_hospital_capacity': 1800,
        'affinity': 3,
    }

    healthcare_optimizer = HealthcareFacilityOptimization(parameters, seed=42)
    instance = healthcare_optimizer.generate_instance()
    solve_status, solve_time, objective_value = healthcare_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")