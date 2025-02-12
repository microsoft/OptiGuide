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
        daily_appointments = [(zone, np.random.uniform(80, 400)) for zone in zones]

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

        # New data: temperature control costs and capacity
        temp_control_costs = {(u, v): np.random.uniform(5.0, 15.0) for u, v in G.edges}
        max_temp = 8  # Maximum allowable temperature in each vehicle
        
        # Store scenarios in instance
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
            'temp_control_costs': temp_control_costs,
            'max_temp': max_temp
        }

    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        daily_appointments = instance['daily_appointments']
        care_scenarios = instance['care_scenarios']
        financial_rewards = instance['financial_rewards']
        travel_costs = instance['travel_costs']
        temp_control_costs = instance['temp_control_costs']
        max_temp = instance['max_temp']

        model = Model("HCAN_Stochastic")

        # Define variables
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        scenario_vars = {(s, node): model.addVar(vtype="B", name=f"NurseShiftScenario{s}_{node}") for s in range(self.no_of_scenarios) for node in G.nodes}
        shift_budget = model.addVar(vtype="C", name="shift_budget")
        daily_appointment_vars = {i: model.addVar(vtype="B", name=f"Appointment_{i}") for i in range(len(daily_appointments))}
        
        # New variable for temperature control
        temperature_control_vars = {f"TempControl{u}_{v}": model.addVar(vtype="B", name=f"TempControl{u}_{v}") for u, v in G.edges}

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
        # Temperature control costs
        objective_expr -= quicksum(temp_control_costs[(u, v)] * temperature_control_vars[f"TempControl{u}_{v}"] for u, v in G.edges)
        
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
        
        # New constraint for temperature control
        for u, v in G.edges:
            model.addCons(
                temperature_control_vars[f"TempControl{u}_{v}"] * max_temp >= self.min_temp,
                name=f"TemperatureControl_{u}_{v}"
            )
        
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 30,
        'max_nodes': 225,
        'zone_prob': 0.17,
        'exclusion_rate': 0.31,
        'shift_hours': 525,
        'no_of_scenarios': 1500,
        'patient_variation': 0.38,
        'time_variation': 0.52,
        'capacity_variation': 0.17,
        'min_temp': 4,
    }
    
    hcan = HCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")