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

class EmergencyFacilityOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_zones > 0
        assert self.min_travel_cost >= 0 and self.max_travel_cost >= self.min_travel_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_capacity > 0 and self.max_facility_capacity >= self.min_facility_capacity

        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        travel_costs = np.random.randint(self.min_travel_cost, self.max_travel_cost + 1, (self.n_facilities, self.n_zones))
        capacities = np.random.randint(self.min_facility_capacity, self.max_facility_capacity + 1, self.n_facilities)
        zone_demands = np.random.randint(1, 10, self.n_zones)

        graph = Graph.barabasi_albert(self.n_facilities, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_risks = np.random.randint(1, 10, size=len(graph.edges))

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
            "zone_demands": zone_demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_risks": edge_risks
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        travel_costs = instance['travel_costs']
        capacities = instance['capacities']
        zone_demands = instance['zone_demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_risks = instance['edge_risks']

        model = Model("EmergencyFacilityOptimization")
        n_facilities = len(facility_costs)
        n_zones = len(travel_costs[0])

        # Decision variables
        MedicalFacility_vars = {f: model.addVar(vtype="B", name=f"MedicalFacility_{f}") for f in range(n_facilities)}
        ZonalAssign_vars = {(f, z): model.addVar(vtype="B", name=f"MedicalFacility_{f}_Zone_{z}") for f in range(n_facilities) for z in range(n_zones)}
        HillEdge_vars = {edge: model.addVar(vtype="B", name=f"HillEdge_{edge[0]}_{edge[1]}") for edge in graph.edges}

        # Objective: minimize the total cost and risk levels
        model.setObjective(
            quicksum(facility_costs[f] * MedicalFacility_vars[f] for f in range(n_facilities)) +
            quicksum(travel_costs[f, z] * ZonalAssign_vars[f, z] for f in range(n_facilities) for z in range(n_zones)) +
            quicksum(edge_risks[i] * HillEdge_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )

        # Constraints: Each zone demand is met by exactly one medical facility
        for z in range(n_zones):
            model.addCons(quicksum(ZonalAssign_vars[f, z] for f in range(n_facilities)) == 1, f"Zone_{z}_Demand")

        # Constraints: Only open medical facilities can serve zones
        for f in range(n_facilities):
            for z in range(n_zones):
                model.addCons(ZonalAssign_vars[f, z] <= MedicalFacility_vars[f], f"MedicalFacility_{f}_Serve_{z}")

        # Constraints: Medical facilities cannot exceed their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(zone_demands[z] * ZonalAssign_vars[f, z] for z in range(n_zones)) <= capacities[f], f"MedicalFacility_{f}_Capacity")

        # Constraints: Medical Facility Graph Cliques (to minimize safety risk levels)
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(MedicalFacility_vars[node] for node in group) <= 1, f"Clique_{count}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 150,
        'n_zones': 112,
        'min_travel_cost': 0,
        'max_travel_cost': 1500,
        'min_facility_cost': 375,
        'max_facility_cost': 5000,
        'min_facility_capacity': 67,
        'max_facility_capacity': 900,
        'affinity': 15,
    }

    emergency_optimizer = EmergencyFacilityOptimization(parameters, seed=42)
    instance = emergency_optimizer.generate_instance()
    solve_status, solve_time, objective_value = emergency_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")