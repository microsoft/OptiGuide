import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GardenMaintenanceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_zone_graph(self):
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        G = nx.erdos_renyi_graph(n=n_zones, p=self.zone_connect_prob, directed=True, seed=self.seed)
        return G

    def generate_task_complexity(self, G):
        for u, v in G.edges:
            G[u][v]['complexity'] = np.random.uniform(self.min_task_complexity, self.max_task_complexity)
        return G

    def generate_zone_data(self, G):
        for node in G.nodes:
            G.nodes[node]['maintenance_proficiency'] = np.random.randint(self.min_proficiency, self.max_proficiency)
        return G

    def get_instance(self):
        G = self.generate_zone_graph()
        G = self.generate_task_complexity(G)
        G = self.generate_zone_data(G)

        max_maintenance_effort = {node: np.random.randint(self.min_effort, self.max_effort) for node in G.nodes}
        neighborhood_benefits = {node: np.random.uniform(self.min_benefit, self.max_benefit) for node in G.nodes}
        task_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        critical_habitats = [set(task) for task in nx.find_cliques(G.to_undirected()) if len(task) <= self.max_habitat_size]
        maintenance_efficiencies = {node: np.random.randint(self.min_efficiency, self.max_efficiency) for node in G.nodes}
        resource_availability = {node: np.random.randint(0, 2) for node in G.nodes}
        
        ### New data generation for added complexity ###
        zone_maintenance_costs = {node: np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost) for node in G.nodes}
        zone_resource_limits = {node: np.random.randint(self.min_resource_limit, self.max_resource_limit) for node in G.nodes}
        zone_capacity = {node: np.random.randint(self.min_capacity, self.max_capacity) for node in G.nodes}
        
        return {
            'G': G,
            'max_maintenance_effort': max_maintenance_effort,
            'neighborhood_benefits': neighborhood_benefits,
            'task_probabilities': task_probabilities,
            'critical_habitats': critical_habitats,
            'maintenance_efficiencies': maintenance_efficiencies,
            'resource_availability': resource_availability,
            'zone_maintenance_costs': zone_maintenance_costs,
            'zone_resource_limits': zone_resource_limits,
            'zone_capacity': zone_capacity,
        }

    def solve(self, instance):
        G = instance['G']
        max_maintenance_effort = instance['max_maintenance_effort']
        neighborhood_benefits = instance['neighborhood_benefits']
        task_probabilities = instance['task_probabilities']
        critical_habitats = instance['critical_habitats']
        maintenance_efficiencies = instance['maintenance_efficiencies']
        resource_availability = instance['resource_availability']
        zone_maintenance_costs = instance['zone_maintenance_costs']
        zone_resource_limits = instance['zone_resource_limits']
        zone_capacity = instance['zone_capacity']

        model = Model("GardenMaintenanceOptimization")

        maintenance_effort_vars = {node: model.addVar(vtype="C", name=f"MaintenanceEffort_{node}") for node in G.nodes}
        zone_task_vars = {(u, v): model.addVar(vtype="B", name=f"ZoneTask_{u}_{v}") for u, v in G.edges}
        habitat_health_vars = {node: model.addVar(vtype="B", name=f"HabitatHealth_{node}") for node in G.nodes}
        resource_allocation_vars = {node: model.addVar(vtype="I", name=f"ResourceAllocation_{node}") for node in G.nodes}
        zone_capacity_vars = {node: model.addVar(vtype="B", name=f"ZoneCapacity_{node}") for node in G.nodes}

        critical_habitat_vars = {}
        for i, habitat in enumerate(critical_habitats):
            critical_habitat_vars[i] = model.addVar(vtype="B", name=f"CriticalHabitat_{i}")

        # Objective: maximizing benefits while minimizing costs and complexity
        objective_expr = quicksum(
            neighborhood_benefits[node] * maintenance_effort_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['complexity'] * zone_task_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr += quicksum(
            maintenance_efficiencies[node] * habitat_health_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            zone_maintenance_costs[node] * zone_capacity_vars[node]
            for node in G.nodes
        )
        objective_expr += quicksum(
            resource_availability[node] * resource_allocation_vars[node]
            for node in G.nodes
        )

        # Constraints
        for node in G.nodes:
            model.addCons(
                maintenance_effort_vars[node] <= max_maintenance_effort[node],
                name=f"MaxMaintenanceEffort_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                zone_task_vars[(u, v)] <= task_probabilities[(u, v)],
                name=f"TaskProbability_{u}_{v}"
            )
            model.addCons(
                zone_task_vars[(u, v)] <= maintenance_effort_vars[u],
                name=f"TaskAssignLimit_{u}_{v}"
            )

        for habitat in critical_habitats:
            model.addCons(
                quicksum(habitat_health_vars[node] for node in habitat) <= 1,
                name=f"MaxOneCriticalHabitat_{[node for node in habitat]}"
            )

        for node in G.nodes:
            model.addCons(
                resource_allocation_vars[node] <= zone_resource_limits[node],
                name=f"ResourceAllocationLimit_{node}"
            )

        for node in G.nodes:
            model.addCons(
                resource_allocation_vars[node] <= zone_capacity[node] * zone_capacity_vars[node],
                name=f"ZoneCapacity_{node}"
            )

        for node in G.nodes:
            model.addCons(
                quicksum(zone_task_vars[(u, v)] for u, v in G.edges if u == node or v == node) <= zone_resource_limits[node],
                name=f"ResourceLimit_{node}"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 15,
        'max_zones': 750,
        'zone_connect_prob': 0.24,
        'min_task_complexity': 364,
        'max_task_complexity': 1350,
        'min_proficiency': 0,
        'max_proficiency': 1250,
        'min_effort': 600,
        'max_effort': 675,
        'min_benefit': 37.5,
        'max_benefit': 2800.0,
        'max_habitat_size': 2100,
        'min_efficiency': 135,
        'max_efficiency': 300,
        'min_maintenance_cost': 555,
        'max_maintenance_cost': 600,
        'min_resource_limit': 10,
        'max_resource_limit': 450,
        'min_capacity': 2,
        'max_capacity': 15,
    }
    
    optimizer = GardenMaintenanceOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")