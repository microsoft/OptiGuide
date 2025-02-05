import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations

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
        visiting_feasibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}
        daily_appointments = [(zone, np.random.uniform(80, 400)) for zone in zones]

        care_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            care_scenarios[s]['patients'] = {node: np.random.normal(G.nodes[node]['patients'], G.nodes[node]['patients'] * self.patient_variation) for node in G.nodes}
            care_scenarios[s]['visit_time'] = {(u, v): np.random.normal(G[u][v]['visit_time'], G[u][v]['visit_time'] * self.time_variation) for u, v in G.edges}
            care_scenarios[s]['healthcare_cap'] = {node: np.random.normal(healthcare_cap[node], healthcare_cap[node] * self.capacity_variation) for node in G.nodes}

        node_existence_prob = np.random.uniform(0.8, 1, len(G.nodes))
        node_weights = np.random.randint(1, 300, len(G.nodes))
        knapsack_capacity = np.random.randint(400, 800)
        cliques_of_interest = [set(zone) for zone in zones if len(zone) <= self.max_clique_size]

        # New data generation for carbon emissions and renewable energy
        carbon_emissions = {(u, v): np.random.uniform(10, 100) for u, v in G.edges}
        renewable_energy = {(u, v): np.random.uniform(0.1, 1.0) for u, v in G.edges}

        # Integration of second MILP's constraints and variables
        traffic_conditions = {(u, v): np.random.uniform(1.0, 3.0) for u, v in G.edges}
        disaster_severity = {node: np.random.uniform(0.5, 2.0) for node in G.nodes}
        emergency_facility_capacities = {node: np.random.randint(1, 5) for node in G.nodes}
        emergency_facility_availability = {node: np.random.choice([0, 1], p=[0.2, 0.8]) for node in G.nodes}
        
        return {
            'G': G,
            'E_invalid': E_invalid,
            'zones': zones,
            'healthcare_cap': healthcare_cap,
            'shift_cost': shift_cost,
            'visiting_feasibility': visiting_feasibility,
            'daily_appointments': daily_appointments,
            'care_scenarios': care_scenarios,
            'node_existence_prob': node_existence_prob,
            'node_weights': node_weights,
            'knapsack_capacity': knapsack_capacity,
            'cliques_of_interest': cliques_of_interest,
            'carbon_emissions': carbon_emissions,
            'renewable_energy': renewable_energy,
            'traffic_conditions': traffic_conditions,
            'disaster_severity': disaster_severity,
            'emergency_facility_capacities': emergency_facility_capacities,
            'emergency_facility_availability': emergency_facility_availability
        }

    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        visiting_feasibility = instance['visiting_feasibility']
        daily_appointments = instance['daily_appointments']
        care_scenarios = instance['care_scenarios']
        node_existence_prob = instance['node_existence_prob']
        node_weights = instance['node_weights']
        knapsack_capacity = instance['knapsack_capacity']
        cliques_of_interest = instance['cliques_of_interest']
        carbon_emissions = instance['carbon_emissions']
        renewable_energy = instance['renewable_energy']
        traffic_conditions = instance['traffic_conditions']
        disaster_severity = instance['disaster_severity']
        emergency_facility_capacities = instance['emergency_facility_capacities']
        emergency_facility_availability = instance['emergency_facility_availability']
        
        model = Model("HCAN")
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        shift_budget = model.addVar(vtype="C", name="shift_budget")
        daily_appointment_vars = {i: model.addVar(vtype="B", name=f"Appointment_{i}") for i in range(len(daily_appointments))}
        patient_vars = {s: {f"NurseShift{node}_s{s}": model.addVar(vtype="B", name=f"NurseShift{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        visit_time_vars = {s: {f"HomeVisit{u}_{v}_s{s}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.no_of_scenarios)}
        care_cap_vars = {s: {f"Capacity{node}_s{s}": model.addVar(vtype="B", name=f"Capacity{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        robust_vars = {f"Robust{node}": model.addVar(vtype="B", name=f"Robust{node}") for node in G.nodes}

        # New variables for carbon emissions and renewable energy
        carbon_emission_vars = {f"CarbonEmission{u}_{v}": model.addVar(vtype="C", name=f"CarbonEmission{u}_{v}") for u, v in G.edges}
        renewable_energy_vars = {f"RenewableEnergy{u}_{v}": model.addVar(vtype="C", name=f"RenewableEnergy{u}_{v}") for u, v in G.edges}
        
        # Integration of new variables for traffic conditions and disaster severity
        traffic_delay_vars = {f"TrafficDelay{u}_{v}": model.addVar(vtype="C", name=f"TrafficDelay{u}_{v}") for u, v in G.edges}
        disaster_response_vars = {f"DisasterResponse{node}": model.addVar(vtype="B", name=f"DisasterResponse{node}") for node in G.nodes}
        
        # Integration of new variables for vehicles
        ambulance_vars = {f"Ambulance{node}": model.addVar(vtype="B", name=f"Ambulance{node}") for node in G.nodes}
        fire_truck_vars = {f"FireTruck{node}": model.addVar(vtype="B", name=f"FireTruck{node}") for node in G.nodes}
        rescue_vehicle_vars = {f"RescueVehicle{node}": model.addVar(vtype="B", name=f"RescueVehicle{node}") for node in G.nodes}

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
        objective_expr -= quicksum(
            visiting_feasibility[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * daily_appointment_vars[i] for i, (bundle, price) in enumerate(daily_appointments))
        objective_expr += quicksum(node_existence_prob[node] * robust_vars[f"Robust{node}"] for node in G.nodes)
        
        # Incorporating new objectives
        objective_expr -= quicksum(
            carbon_emission_vars[f"CarbonEmission{u}_{v}"] for u, v in G.edges
        )
        objective_expr += quicksum(
            renewable_energy_vars[f"RenewableEnergy{u}_{v}"] for u, v in G.edges
        )
        # Including the impact of traffic conditions
        objective_expr -= quicksum(
            traffic_conditions[(u, v)] * traffic_delay_vars[f"TrafficDelay{u}_{v}"] for u, v in G.edges
        )
        # Including the impact of disaster severity
        objective_expr += quicksum(
            disaster_severity[node] * disaster_response_vars[f"DisasterResponse{node}"] for node in G.nodes
        )

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
                nurse_shift_vars[f"NurseShift{u}"] + nurse_shift_vars[f"NurseShift{v}"] >= 2 * home_visit_vars[f"HomeVisit{u}_{v}"],
                name=f"PatientFlow_{u}_{v}_other"
            )

        model.addCons(
            shift_budget <= self.shift_hours,
            name="OffTime_Limit"
        )

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

        # New Constraints for Robust HCAN Problem
        for clique in cliques_of_interest:
            if len(clique) > 1:
                model.addCons(quicksum(robust_vars[f"Robust{node}"] for node in clique) <= 1, name=f"EnhancedClique_{clique}")

        model.addCons(
            quicksum(node_weights[node] * robust_vars[f"Robust{node}"] for node in G.nodes) <= knapsack_capacity,
            name="RobustKnapsack"
        )

        # New constraints for carbon emissions and renewable energy
        model.addCons(
            quicksum(carbon_emission_vars[f"CarbonEmission{u}_{v}"] * home_visit_vars[f"HomeVisit{u}_{v}"] for u, v in G.edges) <= self.carbon_budget,
            name="CarbonBudget"
        )
        for u, v in G.edges:
            model.addCons(
                renewable_energy_vars[f"RenewableEnergy{u}_{v}"] == renewable_energy[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"],
                name=f"RenewableEnergyUtilization_{u}_{v}"
            )
            model.addCons(
                carbon_emission_vars[f"CarbonEmission{u}_{v}"] == carbon_emissions[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"],
                name=f"CarbonEmissionCount_{u}_{v}"
            )
        
        # New constraints for traffic delay
        for u, v in G.edges:
            model.addCons(
                traffic_delay_vars[f"TrafficDelay{u}_{v}"] == traffic_conditions[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"],
                name=f"TrafficDelay_{u}_{v}"
            )
        
        # New constraints for disaster response
        for node in G.nodes:
            model.addCons(
                disaster_response_vars[f"DisasterResponse{node}"] <= disaster_severity[node],
                name=f"DisasterResponse_{node}"
            )

        # New constraints for vehicle capacities
        for node in G.nodes:
            model.addCons(
                quicksum([ambulance_vars[f"Ambulance{node}"], fire_truck_vars[f"FireTruck{node}"], rescue_vehicle_vars[f"RescueVehicle{node}"]]) <= emergency_facility_capacities[node],
                name=f"Capacity_{node}"
            )
            model.addCons(
                quicksum([ambulance_vars[f"Ambulance{node}"], fire_truck_vars[f"FireTruck{node}"], rescue_vehicle_vars[f"RescueVehicle{node}"]]) <= emergency_facility_availability[node],
                name=f"Availability_{node}"
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
        'max_nodes': 765,
        'zone_prob': 0.1,
        'exclusion_rate': 0.1,
        'shift_hours': 420,
        'no_of_scenarios': 126,
        'patient_variation': 0.1,
        'time_variation': 0.31,
        'capacity_variation': 0.59,
        'max_clique_size': 3000,
        'carbon_budget': 10000,
    }
    hcan = HCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")