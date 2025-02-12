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

class HealthcareNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
            
    def gamma_dist(self, shape, scale, size):
        return np.random.gamma(shape, scale, size)

    def generate_instance(self):
        assert self.Number_of_Hospitals > 0 and self.Number_of_ServiceZones > 0
        assert self.Min_Hospital_Cost >= 0 and self.Max_Hospital_Cost >= self.Min_Hospital_Cost
        assert self.ServiceZone_Cost_Lower_Bound >= 0 and self.ServiceZone_Cost_Upper_Bound >= self.ServiceZone_Cost_Lower_Bound
        assert self.Min_Hospital_Capacity > 0 and self.Max_Hospital_Capacity >= self.Min_Hospital_Capacity
        assert self.Min_Operational_Capacity > 0 and self.Max_Hospital_Capacity >= self.Min_Operational_Capacity
        
        hospital_costs = self.gamma_dist(2., 200, self.Number_of_Hospitals).astype(int)
        service_zone_costs = self.gamma_dist(2., 100, (self.Number_of_Hospitals, self.Number_of_ServiceZones)).astype(int)
        hospital_capacities = self.gamma_dist(2., 700, self.Number_of_Hospitals).astype(int)
        service_zone_demands = self.gamma_dist(2., 5, self.Number_of_ServiceZones).astype(int)
        
        graph = Graph.barabasi_albert(self.Number_of_Hospitals, self.Affinity)
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
        for node in range(self.Number_of_Hospitals):
            if node not in used_nodes:
                incompatibilities.add((node,))
        
        # Medical resources and emergency scenarios
        medical_resources = {h: self.gamma_dist(2., 25, 1)[0] for h in range(self.Number_of_Hospitals)}
        emergency_scenarios = [{} for _ in range(self.No_of_EmergencyScenarios)]
        for s in range(self.No_of_EmergencyScenarios):
            emergency_scenarios[s]['demand'] = {z: np.random.normal(service_zone_demands[z], service_zone_demands[z] * self.Demand_Variation) for z in range(self.Number_of_ServiceZones)}

        return {
            "hospital_costs": hospital_costs,
            "service_zone_costs": service_zone_costs,
            "hospital_capacities": hospital_capacities,
            "service_zone_demands": service_zone_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
            "medical_resources": medical_resources,
            "emergency_scenarios": emergency_scenarios
        }

    def solve(self, instance):
        hospital_costs = instance['hospital_costs']
        service_zone_costs = instance['service_zone_costs']
        hospital_capacities = instance['hospital_capacities']
        service_zone_demands = instance['service_zone_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        medical_resources = instance['medical_resources']
        emergency_scenarios = instance['emergency_scenarios']
        
        model = Model("HealthcareNetworkDesign")
        number_of_hospitals = len(hospital_costs)
        number_of_service_zones = len(service_zone_costs[0])

        M = sum(hospital_capacities)  # Big M
        
        # Decision variables
        hospital_vars = {h: model.addVar(vtype="B", name=f"Hospital_{h}") for h in range(number_of_hospitals)}
        service_zone_vars = {(h, z): model.addVar(vtype="B", name=f"Hospital_{h}_ServiceZone_{z}") for h in range(number_of_hospitals) for z in range(number_of_service_zones)}
        resource_binary_vars = {h: model.addVar(vtype="B", name=f"ResourceUsage_{h}") for h in range(number_of_hospitals)}
        medical_resource_util_vars = {h: model.addVar(vtype="C", lb=0.0, ub=medical_resources[h], name=f"MedicalResourceUtil_{h}") for h in range(number_of_hospitals)}
        hospital_operation_vars = {h: model.addVar(vtype="C", lb=0, ub=hospital_capacities[h], name=f"HospitalOperation_{h}") for h in range(number_of_hospitals)}

        # Objective: minimize the total cost including hospital costs, medical resource utilization, and service zone assignment costs
        model.setObjective(
            quicksum(hospital_costs[h] * hospital_vars[h] for h in range(number_of_hospitals)) +
            quicksum(service_zone_costs[h, z] * service_zone_vars[h, z] for h in range(number_of_hospitals) for z in range(number_of_service_zones)) +
            quicksum(medical_resource_util_vars[h] for h in range(number_of_hospitals)),
            "minimize"
        )
        
        # Constraints: Each service zone demand is met by exactly one hospital
        for z in range(number_of_service_zones):
            model.addCons(quicksum(service_zone_vars[h, z] for h in range(number_of_hospitals)) == 1, f"ServiceZone_{z}_Demand")
        
        # Constraints: Only open hospitals can serve service zones
        for h in range(number_of_hospitals):
            for z in range(number_of_service_zones):
                model.addCons(service_zone_vars[h, z] <= hospital_vars[h], f"Hospital_{h}_Serve_{z}")
        
        # Constraints: Hospitals cannot exceed their capacity using Big M
        for h in range(number_of_hospitals):
            model.addCons(hospital_operation_vars[h] <= hospital_capacities[h] * hospital_vars[h], f"Hospital_{h}_Capacity")

        # Constraints: Minimum operational capacity if hospital is open
        for h in range(number_of_hospitals):
            model.addCons(hospital_operation_vars[h] >= self.Min_Operational_Capacity * hospital_vars[h], f"Hospital_{h}_MinOperational")

        # Constraints: Hospital Graph Incompatibilities
        for count, group in enumerate(incompatibilities):
            model.addCons(quicksum(hospital_vars[node] for node in group) <= 1, f"Incompatibility_{count}")

        # Medical resource capacity constraints
        for h in range(number_of_hospitals):
            model.addCons(
                resource_binary_vars[h] * medical_resources[h] >= medical_resource_util_vars[h],
                name=f"ResourceUsage_{h}"
            )
            model.addCons(
                hospital_vars[h] <= resource_binary_vars[h],
                name=f"HospitalResourceUsage_{h}"
            )
            model.addCons(
                medical_resource_util_vars[h] <= medical_resources[h],
                name=f"MaxMedicalResourceUtil_{h}"
            )
        
        # Emergency scenario-based constraints
        for s in range(self.No_of_EmergencyScenarios):
            for h in range(number_of_hospitals):
                for z in range(number_of_service_zones):
                    model.addCons(
                        emergency_scenarios[s]['demand'][z] * service_zone_vars[h, z] <= hospital_capacities[h],
                        name=f"EmergencyScenario_{s}_Hospital_{h}_ServiceZone_{z}"
                    )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Hospitals': 140,
        'Number_of_ServiceZones': 80,
        'ServiceZone_Cost_Lower_Bound': 200,
        'ServiceZone_Cost_Upper_Bound': 2500,
        'Min_Hospital_Cost': 1000,
        'Max_Hospital_Cost': 6000,
        'Min_Hospital_Capacity': 300,
        'Max_Hospital_Capacity': 1500,
        'Min_Operational_Capacity': 500,
        'Affinity': 20,
        'No_of_EmergencyScenarios': 4,
        'Demand_Variation': 0.5,
    }
    
    healthcare_network_optimizer = HealthcareNetworkDesign(parameters, seed=42)
    instance = healthcare_network_optimizer.generate_instance()
    solve_status, solve_time, objective_value = healthcare_network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")