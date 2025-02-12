import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DisasterResponsePlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_zone_graph(self):
        n_nodes = np.random.randint(self.min_units, self.max_units)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.zone_prob, directed=True, seed=self.seed)
        return G

    def generate_zone_hazard(self, G):
        for u, v in G.edges:
            G[u][v]['hazard'] = np.random.uniform(self.min_hazard, self.max_hazard)
        return G

    def generate_unit_data(self, G):
        for node in G.nodes:
            G.nodes[node]['skill'] = np.random.randint(self.min_skill, self.max_skill)
        return G

    def get_instance(self):
        G = self.generate_zone_graph()
        G = self.generate_zone_hazard(G)
        G = self.generate_unit_data(G)

        max_mission_time = {node: np.random.randint(self.min_mission_time, self.max_mission_time) for node in G.nodes}
        zone_importance = {node: np.random.uniform(self.min_importance, self.max_importance) for node in G.nodes}
        zone_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        crucial_zones = [set(zone) for zone in nx.find_cliques(G.to_undirected()) if len(zone) <= self.max_zone_size]
        unit_efficiencies = {node: np.random.randint(self.min_efficiency, self.max_efficiency) for node in G.nodes}
        resource_limits = {node: np.random.randint(1, 3) for node in G.nodes}

        skill_groups = {i: set(random.sample(G.nodes, int(len(G.nodes) / self.num_skill_groups))) for i in range(self.num_skill_groups)}
        
        hazard_exposure = {edge: np.random.normal(5, 2) for edge in G.edges}
        unit_capacity = {node: np.random.randint(5, 20) for node in G.nodes}
        resource_availability = {node: np.random.randint(10, 50) for node in G.nodes}

        # New data for robustness
        hazard_uncertainty = {edge: np.random.uniform(1, 3) for edge in G.edges}
        capacity_variability = {node: np.random.uniform(0.9, 1.1) for node in G.nodes}
        probability_variability = {edge: np.random.uniform(0.8, 1.2) for edge in G.edges}

        return {
            'G': G,
            'max_mission_time': max_mission_time,
            'zone_importance': zone_importance,
            'zone_probabilities': zone_probabilities,
            'crucial_zones': crucial_zones,
            'unit_efficiencies': unit_efficiencies,
            'resource_limits': resource_limits,
            'skill_groups': skill_groups,
            'hazard_exposure': hazard_exposure,
            'unit_capacity': unit_capacity,
            'resource_availability': resource_availability,
            'hazard_uncertainty': hazard_uncertainty,
            'capacity_variability': capacity_variability,
            'probability_variability': probability_variability,
        }

    def solve(self, instance):
        G = instance['G']
        max_mission_time = instance['max_mission_time']
        zone_importance = instance['zone_importance']
        zone_probabilities = instance['zone_probabilities']
        crucial_zones = instance['crucial_zones']
        unit_efficiencies = instance['unit_efficiencies']
        resource_limits = instance['resource_limits']
        skill_groups = instance['skill_groups']
        hazard_exposure = instance['hazard_exposure']
        unit_capacity = instance['unit_capacity']
        resource_availability = instance['resource_availability']
        hazard_uncertainty = instance['hazard_uncertainty']
        capacity_variability = instance['capacity_variability']
        probability_variability = instance['probability_variability']

        model = Model("DisasterResponsePlanning")

        unit_vars = {node: model.addVar(vtype="C", name=f"Unit_{node}") for node in G.nodes}
        zone_vars = {(u, v): model.addVar(vtype="B", name=f"Zone_{u}_{v}") for u, v in G.edges}
        crucial_zone_vars = {node: model.addVar(vtype="B", name=f"Crucial_{node}") for node in G.nodes}
        resource_vars = {node: model.addVar(vtype="B", name=f"Resource_{node}") for node in G.nodes}

        skill_group_vars = {(node, group): model.addVar(vtype="B", name=f"Group_{group}_{node}") 
                      for node in G.nodes for group in skill_groups}
        
        mission_assignment_vars = {(u, v): model.addVar(vtype="B", name=f"Mission_Assignment_{u}_{v}") for u, v in G.edges}
        resource_usage_vars = {node: model.addVar(vtype="C", lb=0, name=f"Resource_Usage_{node}") for node in G.nodes}
        
        M = 1000

        objective_expr = quicksum(
            zone_importance[node] * unit_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            (G[u][v]['hazard'] + hazard_uncertainty[(u, v)]) * zone_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr += quicksum(
            unit_efficiencies[node] * crucial_zone_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            resource_availability[node] * resource_vars[node]
            for node in G.nodes
        )

        for node in G.nodes:
            model.addCons(
                unit_vars[node] <= max_mission_time[node] * capacity_variability[node],
                name=f"MaxMissionTime_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                zone_vars[(u, v)] <= zone_probabilities[(u, v)] * probability_variability[(u, v)],
                name=f"ZoneProbability_{u}_{v}"
            )
            model.addCons(
                zone_vars[(u, v)] <= unit_vars[u],
                name=f"ZoneAssignLimit_{u}_{v}"
            )
        
        for zone in crucial_zones:
            model.addCons(
                quicksum(crucial_zone_vars[node] for node in zone) <= 1,
                name=f"MaxOneCrucialZone_{zone}"
            )

        for node in G.nodes:
            model.addCons(
                resource_vars[node] <= resource_limits[node],
                name=f"ResourceLimit_{node}"
            )

        for group, members in skill_groups.items():
            for zone in crucial_zones:
                model.addCons(
                    quicksum(skill_group_vars[(node, group)] for node in zone if node in members) <= 1,
                    name=f"SetPackingSkillGroup_{group}_Zone_{zone}"
                )

        for node in G.nodes:
            model.addCons(
                quicksum(zone_vars[(u,v)] for u,v in G.edges if u == node or v == node) <= resource_limits[node] * quicksum(zone_probabilities[(u,v)] for u,v in G.edges),
                name=f"KnapsackResourceUtilization_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                quicksum(skill_group_vars[(u, group)] for group in skill_groups if u in skill_groups[group]) >= zone_vars[(u, v)],
                name=f"ZoneCoverage_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                resource_usage_vars[node] <= M * resource_vars[node],
                name=f"ResourceUsageCondition_{node}"
            )
            model.addCons(
                resource_usage_vars[node] >= unit_capacity[node] * resource_vars[node],
                name=f"ResourceLowerBoundCondition_{node}"
            )
            model.addCons(
                resource_usage_vars[node] <= resource_availability[node],
                name=f"ResourceAvailability_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                mission_assignment_vars[(u, v)] <= zone_vars[(u, v)] * M,
                name=f"MissionAssignmentCondition_{u}_{v}"
            )
            for w in G.nodes:
                if w != u and (u, w) in G.edges:
                    model.addCons(
                        mission_assignment_vars[(u, w)] + mission_assignment_vars[(u, v)] <= 1,
                        name=f"UnitCapacityLimit_{u}"
                    )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_units': 50,
        'max_units': 750,
        'zone_prob': 0.1,
        'min_hazard': 100,
        'max_hazard': 900,
        'min_skill': 0,
        'max_skill': 3000,
        'min_mission_time': 1000,
        'max_mission_time': 5000,
        'min_importance': 7.0,
        'max_importance': 1500.0,
        'max_zone_size': 600,
        'min_efficiency': 0,
        'max_efficiency': 750,
        'num_skill_groups': 500,
        'Big_M': 3000,
    }

    optimizer = DisasterResponsePlanning(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")