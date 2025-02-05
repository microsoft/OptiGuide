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

    def add_remote_location_data(self, G):
        remote_prob = 0.3
        for u, v in G.edges:
            G[u][v]['is_remote'] = np.random.choice([0, 1], p=[1-remote_prob, remote_prob])
            G[u][v]['fuel_consumption'] = np.random.uniform(10, 50)
            G[u][v]['load'] = np.random.uniform(1, 10)
            G[u][v]['driving_hours'] = np.random.uniform(5, 20)

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_healthcare_data(G)
        self.add_remote_location_data(G)
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

        # Generate mutual exclusivity groups for visits
        mutual_exclusivity_groups = []
        edge_list = list(G.edges)
        for _ in range(self.n_exclusive_groups):
            group = random.sample(range(len(edge_list)), self.group_size)
            mutual_exclusivity_groups.append([edge_list[idx] for idx in group])

        return {
            'G': G,
            'E_invalid': E_invalid,
            'zones': zones,
            'healthcare_cap': healthcare_cap,
            'shift_cost': shift_cost,
            'care_scenarios': care_scenarios,
            'mutual_exclusivity_groups': mutual_exclusivity_groups
        }
    
    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        care_scenarios = instance['care_scenarios']
        mutual_exclusivity_groups = instance['mutual_exclusivity_groups']

        model = Model("HCAN")
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        shift_budget = model.addVar(vtype="C", name="shift_budget")
        is_remote = {(u, v): model.addVar(vtype="B", name=f"is_remote_{u}_{v}") for u, v in G.edges}
        
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

        # New Constraints
        # 40% routes must prioritize remote locations
        model.addCons(
            quicksum(is_remote[u, v] for u, v in G.edges) >= 0.4 * quicksum(home_visit_vars[f"HomeVisit{u}_{v}"] for u, v in G.edges),
            name="RemoteRoute_Minimum_40%"
        )

        # Fuel consumption for remote deliveries <= 25% of total fuel budget
        total_fuel_budget = quicksum(G[u][v]['fuel_consumption'] * home_visit_vars[f"HomeVisit{u}_{v}"] for u, v in G.edges)
        remote_fuel_budget = quicksum(G[u][v]['fuel_consumption'] * is_remote[u, v] for u, v in G.edges)
        model.addCons(
            remote_fuel_budget <= 0.25 * total_fuel_budget,
            name="RemoteFuel_Budget_25%"
        )

        # Load capacity constraints for each vehicle
        for u, v in G.edges:
            model.addCons(
                G[u][v]['load'] * home_visit_vars[f"HomeVisit{u}_{v}"] <= 1.15 * self.vehicle_capacity,
                name=f"Vehicle_Load_Capacity_{u}_{v}"
            )

        # Driving hours constraints
        for u, v in G.edges:
            model.addCons(
                G[u][v]['driving_hours'] * home_visit_vars[f"HomeVisit{u}_{v}"] <= 0.2 * self.regular_working_hours,
                name=f"Driving_Hours_{u}_{v}"
            )

        # Mutual exclusivity groups - Big M constraints
        M_mutual_exclusive = 2  # Upper bound for the maximum number of selected visits in a group.
        z_vars = {i: model.addVar(vtype="C", name=f"Z_{i}") for i in range(len(mutual_exclusivity_groups))}

        for i, group in enumerate(mutual_exclusivity_groups):
            model.addCons(
                quicksum(home_visit_vars[f"HomeVisit{u}_{v}"] for u, v in group) <= z_vars[i],
                name=f"MutualExclusivityGroup_{i}_Upper"
            )
            model.addCons(
                z_vars[i] <= M_mutual_exclusive,
                name=f"MutualExclusivityGroup_{i}_Lower"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 33,
        'max_nodes': 803,
        'zone_prob': 0.1,
        'exclusion_rate': 0.52,
        'shift_hours': 1680,
        'no_of_scenarios': 63,
        'patient_variation': 0.73,
        'time_variation': 0.17,
        'capacity_variation': 0.38,
        'vehicle_capacity': 50,
        'regular_working_hours': 480,
        'n_exclusive_groups': 100,
        'group_size': 3,
    }
    
    hcan = HCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")