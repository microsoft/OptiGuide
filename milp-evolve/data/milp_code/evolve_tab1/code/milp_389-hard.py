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

    def create_zones(self, G):
        zones = list(nx.find_cliques(G.to_undirected()))
        return zones

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_healthcare_data(G)
        zones = self.create_zones(G)

        healthcare_cap = {node: np.random.randint(20, 100) for node in G.nodes}
        shift_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}
        daily_appointments = [(zone, np.random.uniform(80, 400)) for zone in zones]

        return {
            'G': G,
            'zones': zones,
            'healthcare_cap': healthcare_cap,
            'shift_cost': shift_cost,
            'daily_appointments': daily_appointments,
        }

    def solve(self, instance):
        G, zones = instance['G'], instance['zones']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        daily_appointments = instance['daily_appointments']

        model = Model("HCAN")
        nurse_shift_vars = {f"NurseShift{node}": model.addVar(vtype="B", name=f"NurseShift{node}") for node in G.nodes}
        home_visit_vars = {f"HomeVisit{u}_{v}": model.addVar(vtype="B", name=f"HomeVisit{u}_{v}") for u, v in G.edges}
        daily_appointment_vars = {i: model.addVar(vtype="B", name=f"Appointment_{i}") for i in range(len(daily_appointments))}

        objective_expr = quicksum(
            G.nodes[node]['patients'] * nurse_shift_vars[f"NurseShift{node}"]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            shift_cost[(u, v)] * home_visit_vars[f"HomeVisit{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * daily_appointment_vars[i] for i, (bundle, price) in enumerate(daily_appointments))

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

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 60,
        'max_nodes': 800,
        'zone_prob': 0.17,
        'shift_hours': 1400,
    }

    hcan = HCAN(parameters, seed=seed)
    instance = hcan.get_instance()
    solve_status, solve_time = hcan.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")