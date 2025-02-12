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
        assert self.Number_of_Facilities > 0 and self.Number_of_Regions > 0
        assert self.Min_Facility_Cost >= 0 and self.Max_Facility_Cost >= self.Min_Facility_Cost
        assert self.Region_Cost_Lower_Bound >= 0 and self.Region_Cost_Upper_Bound >= self.Region_Cost_Lower_Bound
        assert self.Min_Facility_Capacity > 0 and self.Max_Facility_Capacity >= self.Min_Facility_Capacity
        assert self.Min_Serve_Cost >= 0 and self.Max_Serve_Cost >= self.Min_Serve_Cost

        facility_costs = np.random.randint(self.Min_Facility_Cost, self.Max_Facility_Cost + 1, self.Number_of_Facilities)
        region_costs = np.random.randint(self.Region_Cost_Lower_Bound, self.Region_Cost_Upper_Bound + 1, (self.Number_of_Facilities, self.Number_of_Regions))
        facility_capacities = np.random.randint(self.Min_Facility_Capacity, self.Max_Facility_Capacity + 1, self.Number_of_Facilities)
        region_demands = np.random.randint(1, 10, self.Number_of_Regions)
        serve_costs = np.random.uniform(self.Min_Serve_Cost, self.Max_Serve_Cost, self.Number_of_Facilities)
        
        graph = Graph.barabasi_albert(self.Number_of_Facilities, self.Affinity)
        cliques = graph.efficient_greedy_clique_partition()
        incompatibilities = set(graph.edges)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                incompatibilities.remove(edge)
            if len(clique) > 1:
                incompatibilities.add(clique)

        used_nodes = set()
        for group in incompatibilities:
            used_nodes.update(group)
        for node in range(self.Number_of_Facilities):
            if node not in used_nodes:
                incompatibilities.add((node,))
        
        # New data elements for pollution control
        pollution_levels = np.random.uniform(0, self.Max_Pollution_Level, self.Number_of_Facilities)
        pollution_deviation = np.random.uniform(0, 1, self.Number_of_Facilities)
        hazard_threshold = np.random.uniform(self.Min_Hazard_Threshold, self.Max_Hazard_Threshold)

        return {
            "facility_costs": facility_costs,
            "region_costs": region_costs,
            "facility_capacities": facility_capacities,
            "region_demands": region_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
            "edge_weights": edge_weights,
            "serve_costs": serve_costs,
            "Big_M": max(region_demands) * self.Max_Facility_Capacity,
            "pollution_levels": pollution_levels,
            "pollution_deviation": pollution_deviation,
            "hazard_threshold": hazard_threshold
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        region_costs = instance['region_costs']
        facility_capacities = instance['facility_capacities']
        region_demands = instance['region_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        edge_weights = instance['edge_weights']
        serve_costs = instance['serve_costs']
        Big_M = instance['Big_M']
        pollution_levels = instance['pollution_levels']
        pollution_deviation = instance['pollution_deviation']
        hazard_threshold = instance['hazard_threshold']
        
        model = Model("HealthcareFacilityOptimization")
        number_of_facilities = len(facility_costs)
        number_of_regions = len(region_costs[0])
        
        # Decision Variables
        facility_vars = {f: model.addVar(vtype="B", name=f"HealthcareFacility_{f}") for f in range(number_of_facilities)}
        region_vars = {(f, r): model.addVar(vtype="B", name=f"Facility_{f}_Region_{r}") for f in range(number_of_facilities) for r in range(number_of_regions)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        service_cost_vars = {f: model.addVar(vtype="C", lb=0, name=f"FacilityServe_{f}") for f in range(number_of_facilities)}
        capacity_vars = {(f, r): model.addVar(vtype="C", lb=0, name=f"Capacity_{f}_Region_{r}") for f in range(number_of_facilities) for r in range(number_of_regions)}
        
        # Pollution control variables
        pollution_vars = {f: model.addVar(vtype="C", lb=0, name=f"Pollution_{f}") for f in range(number_of_facilities)}
        
        # Objective: minimize the total cost including service costs and facility opening costs
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(number_of_facilities)) +
            quicksum(region_costs[f, r] * region_vars[f, r] for f in range(number_of_facilities) for r in range(number_of_regions)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)) +
            quicksum(service_cost_vars[f] for f in range(number_of_facilities)), "minimize"
        )
        
        # Constraints: Each region is served by exactly one healthcare facility
        for r in range(number_of_regions):
            model.addCons(quicksum(region_vars[f, r] for f in range(number_of_facilities)) == 1, f"Region_{r}_Serve")
        
        # Constraints: Only open healthcare facilities can serve regions
        for f in range(number_of_facilities):
            for r in range(number_of_regions):
                model.addCons(region_vars[f, r] <= facility_vars[f], f"Facility_{f}_Serve_{r}")
        
        # Big M Formulation for Capacity Constraints
        for f in range(number_of_facilities):
            for r in range(number_of_regions):
                model.addCons(capacity_vars[f, r] <= Big_M * region_vars[f, r], f"Big_M_{f}_{r}")
                model.addCons(capacity_vars[f, r] <= facility_capacities[f] * region_vars[f, r], f"Capacity_{f}_{r}")
            model.addCons(quicksum(capacity_vars[f, r] for r in range(number_of_regions)) <= facility_capacities[f], f"Total_Capacity_{f}")

        # Constraints: Service Incompatibilities
        for count, group in enumerate(incompatibilities):
            model.addCons(quicksum(facility_vars[node] for node in group) <= 1, f"Incompatibility_{count}")

        # Compatibility Constraints: Prohibit incompatible facilities servicing the same regions
        for i, neighbors in graph.neighbors.items():
            for neighbor in neighbors:
                model.addCons(facility_vars[i] + facility_vars[neighbor] <= 1, f"Neighbor_{i}_{neighbor}")

        # Service cost constraints
        for f in range(number_of_facilities):
            model.addCons(service_cost_vars[f] == serve_costs[f] * quicksum(capacity_vars[f, r] for r in range(number_of_regions)), f"Facility_{f}_ServeCost")
        
        # Pollution control constraints
        for f in range(number_of_facilities):
            model.addCons(pollution_vars[f] == (pollution_levels[f] + pollution_deviation[f]) * facility_vars[f], f"Pollution_Level_{f}")

        # Combined pollution constraints
        model.addCons(quicksum(pollution_vars[f] for f in range(number_of_facilities)) <= hazard_threshold, f"Hazard_Threshold_Constraint")
        
        # Optimization
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Facilities': 100,
        'Number_of_Regions': 150,
        'Region_Cost_Lower_Bound': 5,
        'Region_Cost_Upper_Bound': 1875,
        'Min_Facility_Cost': 250,
        'Max_Facility_Cost': 4500,
        'Min_Facility_Capacity': 225,
        'Max_Facility_Capacity': 600,
        'Affinity': 15,
        'Min_Serve_Cost': 0.45,
        'Max_Serve_Cost': 0.66,
        'Max_Pollution_Level': 75,
        'Min_Hazard_Threshold': 200,
        'Max_Hazard_Threshold': 500,
    }
    
    healthcare_optimizer = HealthcareFacilityOptimization(parameters, seed=42)
    instance = healthcare_optimizer.generate_instance()
    solve_status, solve_time, objective_value = healthcare_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")