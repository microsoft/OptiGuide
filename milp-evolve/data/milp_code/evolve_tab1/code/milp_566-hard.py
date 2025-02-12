import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedHCAN:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.barabasi_albert_graph(n=n_nodes, m=5, seed=self.seed)
        return G

    def generate_healthcare_data(self, G):
        for node in G.nodes:
            G.nodes[node]['patients'] = np.random.poisson(lam=15)
        for u, v in G.edges:
            G[u][v]['visit_time'] = np.random.uniform(1, 5)
            G[u][v]['capacity'] = np.random.poisson(lam=10)

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_zones(self, G):
        zones = list(nx.find_cliques(G))
        return zones

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_healthcare_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        zones = self.create_zones(G)

        healthcare_cap = {node: np.random.poisson(lam=50) for node in G.nodes}
        shift_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}
        daily_appointments = [(zone, np.random.uniform(80, 400)) for zone in zones]

        care_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            care_scenarios[s]['patients'] = {node: np.random.poisson(lam=G.nodes[node]['patients'])
                                              for node in G.nodes}
            care_scenarios[s]['visit_time'] = {(u, v): np.random.uniform(1, 5)
                                               for u, v in G.edges}
            care_scenarios[s]['healthcare_cap'] = {node: np.random.poisson(lam=healthcare_cap[node])
                                                   for node in G.nodes}
        
        financial_rewards = {node: np.random.uniform(10, 100) for node in G.nodes}
        travel_costs = {(u, v): np.random.uniform(1.0, 15.0) for u, v in G.edges}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'zones': zones,
            'healthcare_cap': healthcare_cap,
            'shift_cost': shift_cost,
            'daily_appointments': daily_appointments,
            'care_scenarios': care_scenarios,
            'financial_rewards': financial_rewards,
            'travel_costs': travel_costs,
        }

    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        daily_appointments = instance['daily_appointments']
        care_scenarios = instance['care_scenarios']
        financial_rewards = instance['financial_rewards']
        travel_costs = instance['travel_costs']

        model = Model("Enhanced_HCAN_Stochastic")

        # Define variables
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        scenario_vars = {(s, node): model.addVar(vtype="C", name=f"NurseShiftScenario{s}_{node}") for s in range(self.no_of_scenarios) for node in G.nodes}
        ward_assignment_vars = {f"Ward{node}": model.addVar(vtype="I", lb=0, ub=3, name=f"Ward{node}") for node in G.nodes}
        shift_budget = model.addVar(vtype="C", name="shift_budget")
        daily_appointment_vars = {i: model.addVar(vtype="B", name=f"Appointment_{i}") for i in range(len(daily_appointments))}
        resource_limits = {node: model.addVar(vtype="I", lb=0, ub=100, name=f"Resources{node}") for node in G.nodes}
        
        # Objective function - maximizing the expected number of treated patients minus costs
        objective_expr = quicksum(
            care_scenarios[s]['patients'][node] * scenario_vars[(s, node)]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            shift_cost[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * daily_appointment_vars[i] for i, (bundle, price) in enumerate(daily_appointments))
        objective_expr += quicksum(financial_rewards[node] * nurse_shift_vars[f"NurseShift{node}"] for node in G.nodes)
        objective_expr -= quicksum(travel_costs[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"] for u, v in G.edges)

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
            shift_budget <= sum(resource_limits.values()),
            name="ResourceLimit"
        )

        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    scenario_vars[(s, node)] <= nurse_shift_vars[f"NurseShift{node}"],
                    name=f"ScenarioNurseShift_{s}_{node}"
                )
        
        for node in G.nodes:
            model.addCons(
                ward_assignment_vars[f"Ward{node}"] <= healthcare_cap[node],
                name=f"WardCap_{node}"
            )
        
        clique_assign_vars = {f"CliqueAssign{i}": model.addVar(vtype="B", name=f"CliqueAssign{i}") for i in range(len(zones))}
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(nurse_shift_vars[f"NurseShift{node}"] for node in zone) <= self.max_clique_assignments,
                name=f"CliqueConstraint_{i}"
            )
            
        # Additional diverse constraints
        for u, v in G.edges:
            model.addCons(
                ward_assignment_vars[f"Ward{u}"] + ward_assignment_vars[f"Ward{v}"] - home_visit_vars[f"HomeVisit{u}_{v}"] <= 2,
                name=f"WardVisit_{u}_{v}"
            )
        
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 300,
        'max_nodes': 1000,
        'zone_prob': 0.63,
        'exclusion_rate': 0.78,
        'shift_hours': 3000,
        'no_of_scenarios': 2000,
        'patient_variation': 0.22,
        'time_variation': 0.3,
        'capacity_variation': 0.32,
        'max_clique_assignments': 10,
    }

    hcan = EnhancedHCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")