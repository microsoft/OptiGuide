import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HCAN:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.zone_prob, directed=True, seed=self.seed)
        return G

    def generate_healthcare_data(self, G):
        for node in G.nodes:
            G.nodes[node]['patients'] = np.random.randint(10, 200)
        for u, v in G.edges:
            G[u][v]['visit_time'] = np.random.randint(1, 3)
            G[u][v]['capacity'] = np.random.randint(5, 15)

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_zones(self, G):
        zones = list(nx.find_cliques(G.to_undirected()))
        return zones

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_healthcare_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        zones = self.create_zones(G)

        healthcare_cap = {node: np.random.randint(20, 100) for node in G.nodes}
        shift_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}

        care_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            care_scenarios[s]['patients'] = {node: np.random.normal(G.nodes[node]['patients'], G.nodes[node]['patients'] * self.patient_variation)
                                              for node in G.nodes}
            care_scenarios[s]['visit_time'] = {(u, v): np.random.normal(G[u][v]['visit_time'], G[u][v]['visit_time'] * self.time_variation)
                                               for u, v in G.edges}
            care_scenarios[s]['healthcare_cap'] = {node: np.random.normal(healthcare_cap[node], healthcare_cap[node] * self.capacity_variation)
                                                   for node in G.nodes}
        
        financial_rewards = {node: np.random.uniform(10, 100) for node in G.nodes}
        travel_costs = {(u, v): np.random.uniform(1.0, 10.0) for u, v in G.edges}
        node_existence_prob = np.random.uniform(0.8, 1, len(G.nodes))
        node_weights = np.random.uniform(1, self.max_weight, len(G.nodes))
        knapsack_capacity = np.random.uniform(self.min_capacity, self.max_capacity)
        flow_capacities = {edge: np.random.uniform(1, self.max_flow_capacity) for edge in G.edges}
        NodeResidues = np.random.uniform(1, self.max_residue, len(G.nodes))
        NodeResidues_deviation = np.random.uniform(0, 1, len(G.nodes))
        HazardThreshold = np.random.uniform(self.min_threshold, self.max_threshold)

        return {
            'G': G,
            'E_invalid': E_invalid,
            'zones': zones,
            'healthcare_cap': healthcare_cap,
            'shift_cost': shift_cost,
            'care_scenarios': care_scenarios,
            'financial_rewards': financial_rewards,
            'travel_costs': travel_costs,
            'node_existence_prob': node_existence_prob,
            'node_weights': node_weights,
            'knapsack_capacity': knapsack_capacity,
            'flow_capacities': flow_capacities,
            'NodeResidues': NodeResidues,
            'NodeResidues_deviation': NodeResidues_deviation,
            'HazardThreshold': HazardThreshold,
        }

    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        care_scenarios = instance['care_scenarios']
        financial_rewards = instance['financial_rewards']
        travel_costs = instance['travel_costs']
        node_existence_prob = instance['node_existence_prob']
        node_weights = instance['node_weights']
        knapsack_capacity = instance['knapsack_capacity']
        flow_capacities = instance['flow_capacities']
        NodeResidues = instance['NodeResidues']
        NodeResidues_deviation = instance['NodeResidues_deviation']
        HazardThreshold = instance['HazardThreshold']

        model = Model("HCAN_Complex")

        # Define variables
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        scenario_vars = {(s, node): model.addVar(vtype="B", name=f"NurseShiftScenario{s}_{node}") for s in range(self.no_of_scenarios) for node in G.nodes}
        shift_budget = model.addVar(vtype="C", name="shift_budget")

        flow = {}
        for (i, j) in G.edges:
            flow[i, j] = model.addVar(vtype="C", name=f"flow_{i}_{j}")
            flow[j, i] = model.addVar(vtype="C", name=f"flow_{j}_{i}")
        
        # Objective function
        objective_expr = quicksum(
            care_scenarios[s]['patients'][node] * scenario_vars[(s, node)]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            shift_cost[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(financial_rewards[node] * nurse_shift_vars[f"NurseShift{node}"] for node in G.nodes)
        objective_expr -= quicksum(travel_costs[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"] for u, v in G.edges)
        penalties = quicksum((1 - node_existence_prob[node]) * nurse_shift_vars[f"NurseShift{node}"] for node in G.nodes)
        objective_expr += quicksum(node_existence_prob[node] * nurse_shift_vars[f"NurseShift{node}"] for node in G.nodes) - penalties

        # Constraints
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(nurse_shift_vars[f"NurseShift{node}"] for node in zone) <= 1,
                name=f"WorkerGroup_{i}"
            )
        
        for u, v in G.edges:
            model.addCons(
                nurse_shift_vars[f"NurseShift{u}"] + nurse_shift_vars[f"NurseShift{v}"] <= 1 + home_visit_vars[f"HomeVisit{u}_{v}"],
                name=f"PatientFlow_{u}_{v}"
            )
            
        model.addCons(
            shift_budget <= self.shift_hours,
            name="OffTime_Limit"
        )

        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    scenario_vars[(s, node)] <= nurse_shift_vars[f"NurseShift{node}"],
                    name=f"ScenarioNurseShift_{s}_{node}"
                )

        # Additional Constraints from the second MILP
        model.addCons(
            quicksum(node_weights[node] * nurse_shift_vars[f"NurseShift{node}"] for node in G.nodes) <= knapsack_capacity,
            name="knapsack_constraint"
        )

        pollution_constraint = quicksum((NodeResidues[node] + NodeResidues_deviation[node]) * nurse_shift_vars[f"NurseShift{node}"] for node in G.nodes)
        model.addCons(pollution_constraint <= HazardThreshold, name="pollution_control")

        M = self.max_weight
        for u, v in G.edges:
            y = model.addVar(vtype="B", name=f"y_{u}_{v}")
            model.addCons(nurse_shift_vars[f"NurseShift{u}"] + nurse_shift_vars[f"NurseShift{v}"] - 2 * y <= 0, name=f"bigM1_{u}_{v}")
            model.addCons(nurse_shift_vars[f"NurseShift{u}"] + nurse_shift_vars[f"NurseShift{v}"] + M * (y - 1) >= 0, name=f"bigM2_{u}_{v}")

        for node in G.nodes:
            model.addCons(
                quicksum(flow[i, j] for (i, j) in G.edges if j == node) == quicksum(flow[i, j] for (i, j) in G.edges if i == node),
                name=f"flow_conservation_{node}"
            )

        for (i, j) in G.edges:
            model.addCons(flow[i, j] <= flow_capacities[(i, j)], name=f"flow_capacity_{i}_{j}")
            model.addCons(flow[j, i] <= flow_capacities[(i, j)], name=f"flow_capacity_{j}_{i}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 90,
        'max_nodes': 900,
        'zone_prob': 0.17,
        'exclusion_rate': 0.73,
        'shift_hours': 1180,
        'no_of_scenarios': 1250,
        'patient_variation': 0.17,
        'time_variation': 0.45,
        'capacity_variation': 0.38,
        'max_weight': 1215,
        'min_capacity': 10000,
        'max_capacity': 15000,
        'max_clique_size': 2250,
        'max_flow_capacity': 2109,
        'max_residue': 2020,
        'min_threshold': 10000,
        'max_threshold': 15000,
    }

    hcan = HCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")