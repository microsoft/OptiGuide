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

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'zones': zones, 
            'healthcare_cap': healthcare_cap, 
            'shift_cost': shift_cost,
            'care_scenarios': care_scenarios
        }
    
    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        care_scenarios = instance['care_scenarios']

        model = Model("HCAN")
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        shift_budget = model.addVar(vtype="C", name="shift_budget")

        # Scenario-specific variables
        patient_vars = {s: {f"NurseShift{node}_s{s}": model.addVar(vtype="B", name=f"NurseShift{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        visit_time_vars = {s: {f"HomeVisit{u}_{v}_s{s}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.no_of_scenarios)}
        care_cap_vars = {s: {f"Capacity{node}_s{s}": model.addVar(vtype="B", name=f"Capacity{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}

        objective_expr = quicksum(
            care_scenarios[s]['patients'][node] * patient_vars[s][f"NurseShift{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            care_scenarios[s]['visit_time'][(u, v)] * visit_time_vars[s][f"HomeVisit{u}_{v}_s{s}"]
            for s in range(self.no_of_scenarios) for u, v in E_invalid
        )

        objective_expr -= quicksum(
            care_scenarios[s]['healthcare_cap'][node] * care_scenarios[s]['patients'][node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            shift_cost[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"]
            for u, v in G.edges
        )

        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(nurse_shift_vars[f"NurseShift{node}"] for node in zone) <= 1,
                name=f"WorkerGroup_{i}"
            )

        M = 1000  # Big M constant, set contextually larger than any decision boundary.

        for u, v in G.edges:
            model.addCons(
                nurse_shift_vars[f"NurseShift{u}"] + nurse_shift_vars[f"NurseShift{v}"] <= 1 + M * (1 - home_visit_vars[f"HomeVisit{u}_{v}"]),
                name=f"PatientFlow_{u}_{v}"
            )
            model.addCons(
                nurse_shift_vars[f"NurseShift{u}"] + nurse_shift_vars[f"NurseShift{v}"] >= 2 * home_visit_vars[f"HomeVisit{u}_{v}"] - M * (1 - home_visit_vars[f"HomeVisit{u}_{v}"]),
                name=f"PatientFlow_{u}_{v}_other"
            )

        model.addCons(
            shift_budget <= self.shift_hours,
            name="OffTime_Limit"
        )

        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    patient_vars[s][f"NurseShift{node}_s{s}"] == nurse_shift_vars[f"NurseShift{node}"],
                    name=f"PatientDemandScenario_{node}_s{s}"
                )
                model.addCons(
                    care_cap_vars[s][f"Capacity{node}_s{s}"] == nurse_shift_vars[f"NurseShift{node}"],
                    name=f"ResourceAvailability_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    visit_time_vars[s][f"HomeVisit{u}_{v}_s{s}"] == home_visit_vars[f"HomeVisit{u}_{v}"],
                    name=f"FlowConstraintVisits_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 66,
        'max_nodes': 1071,
        'zone_prob': 0.1,
        'exclusion_rate': 0.8,
        'shift_hours': 840,
        'no_of_scenarios': 84,
        'patient_variation': 0.52,
        'time_variation': 0.38,
        'capacity_variation': 0.59
    }

    hcan = HCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")