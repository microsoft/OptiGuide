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

        # Adjust travel costs to introduce stochasticity
        travel_costs_uncertainty = {(u, v): np.multiply(val, 1 + np.random.normal(0, self.travel_cost_variation)) for (u, v), val in travel_costs.items()}
        
        return {
            'G': G,
            'E_invalid': E_invalid,
            'zones': zones,
            'healthcare_cap': healthcare_cap,
            'shift_cost': shift_cost,
            'care_scenarios': care_scenarios,
            'financial_rewards': financial_rewards,
            'travel_costs': travel_costs,
            'travel_costs_uncertainty': travel_costs_uncertainty
        }

    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        care_scenarios = instance['care_scenarios']
        financial_rewards = instance['financial_rewards']
        travel_costs = instance['travel_costs']
        travel_costs_uncertainty = instance['travel_costs_uncertainty']

        model = Model("HCAN_Simplified")

        # Define variables
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        scenario_vars = {(s, node): model.addVar(vtype="B", name=f"NurseShiftScenario{s}_{node}") for s in range(self.no_of_scenarios) for node in G.nodes}
        shift_budget = model.addVar(vtype="C", name="shift_budget")
        robust_budget = model.addVar(vtype="C", name="robust_budget")
        
        # Objective function - including modeled uncertainties
        objective_expr = quicksum(
            care_scenarios[s]['patients'][node] * scenario_vars[(s, node)]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            shift_cost[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(financial_rewards[node] * nurse_shift_vars[f"NurseShift{node}"] for node in G.nodes)
        objective_expr -= quicksum(travel_costs_uncertainty[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"] for u, v in G.edges)

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
        
        # Robust constraints
        model.addCons(
            robust_budget == shift_budget * (1 + self.robust_factor),
            name="Robust_Budget_Constraint"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 22,
        'max_nodes': 225,
        'zone_prob': 0.24,
        'exclusion_rate': 0.73,
        'shift_hours': 1180,
        'no_of_scenarios': 2500,
        'patient_variation': 0.1,
        'time_variation': 0.45,
        'capacity_variation': 0.8,
        'travel_cost_variation': 0.1,
        'robust_factor': 0.31,
    }
    hcan = HCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")