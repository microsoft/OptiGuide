import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, probability, seed=None):
        graph = nx.erdos_renyi_graph(number_of_nodes, probability, seed=seed)
        edges = set(graph.edges)
        degrees = [d for (n, d) in graph.degree]
        neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
        return Graph(number_of_nodes, edges, degrees, neighbors)

class HospitalResourceManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        doctor_costs = np.random.randint(self.min_doctor_cost, self.max_doctor_cost + 1, self.n_shifts)
        nurse_costs = np.random.randint(self.min_nurse_cost, self.max_nurse_cost + 1, (self.n_shifts, self.n_departments))
        room_capacities = np.random.randint(self.min_room_capacity, self.max_room_capacity + 1, self.n_shifts)
        medical_equipment = np.random.gamma(2, 2, self.n_departments).astype(int) + 1

        graph = Graph.erdos_renyi(self.n_shifts, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        transportation_costs = np.random.randint(1, 20, (self.n_shifts, self.n_shifts))
        transportation_capacities = np.random.randint(self.transportation_capacity_min, self.transportation_capacity_max + 1, (self.n_shifts, self.n_shifts))

        room_transport_costs = np.random.randint(10, 50, (self.n_shifts, self.n_shifts))
        efficiency_costs = np.random.randint(5, 30, (self.n_shifts, self.n_shifts))
        doctor_penalties = np.random.uniform(low=0.5, high=2.0, size=(self.n_shifts, self.n_shifts))

        compliance_costs = np.random.randint(10, 1000, self.n_departments)
        
        hospital_impact = np.random.uniform(0.1, 10, (self.n_shifts, self.n_shifts))

        return {
            "doctor_costs": doctor_costs,
            "nurse_costs": nurse_costs,
            "room_capacities": room_capacities,
            "medical_equipment": medical_equipment,
            "graph": graph,
            "edge_weights": edge_weights,
            "transportation_costs": transportation_costs,
            "transportation_capacities": transportation_capacities,
            "room_transport_costs": room_transport_costs,
            "efficiency_costs": efficiency_costs,
            "doctor_penalties": doctor_penalties,
            "compliance_costs": compliance_costs,
            "hospital_impact": hospital_impact,
        }

    def solve(self, instance):
        doctor_costs = instance['doctor_costs']
        nurse_costs = instance['nurse_costs']
        room_capacities = instance['room_capacities']
        medical_equipment = instance['medical_equipment']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        transportation_costs = instance['transportation_costs']
        transportation_capacities = instance['transportation_capacities']
        room_transport_costs = instance['room_transport_costs']
        efficiency_costs = instance['efficiency_costs']
        doctor_penalties = instance['doctor_penalties']
        compliance_costs = instance['compliance_costs']
        hospital_impact = instance['hospital_impact']

        model = Model("HospitalResourceManagement")
        n_shifts = len(doctor_costs)
        n_departments = len(nurse_costs[0])

        doctor_vars = {h: model.addVar(vtype="B", name=f"Doctor_Scheduling_{h}") for h in range(n_shifts)}
        nurse_vars = {(h, d): model.addVar(vtype="B", name=f"Shift_{h}_Department_{d}") for h in range(n_shifts) for d in range(n_departments)}
        equipment_usage_vars = {(i, j): model.addVar(vtype="I", name=f"Equipment_Usage_{i}_{j}") for i in range(n_shifts) for j in range(n_shifts)}
        equipment_transport_vars = {(i, j): model.addVar(vtype="I", name=f"Transport_{i}_{j}") for i in range(n_shifts) for j in range(n_shifts)}

        room_usage_vars = {(i, j): model.addVar(vtype="C", name=f"Room_Usage_{i}_{j}") for i in range(n_shifts) for j in range(n_shifts)}

        department_vars = {i: model.addVar(vtype="B", name=f"Department_{i}") for i in range(n_shifts)}

        model.setObjective(
            quicksum(doctor_costs[h] * doctor_vars[h] for h in range(n_shifts)) +
            quicksum(nurse_costs[h, d] * nurse_vars[h, d] for h in range(n_shifts) for d in range(n_departments)) +
            quicksum(transportation_costs[i, j] * equipment_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(room_transport_costs[i, j] * equipment_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(efficiency_costs[i, j] * equipment_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(doctor_penalties[i, j] * equipment_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(compliance_costs[d] * nurse_vars[h, d] for h in range(n_shifts) for d in range(n_departments)) +
            quicksum(hospital_impact[i, j] * equipment_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)),
            "minimize"
        )

        for d in range(n_departments):
            model.addCons(quicksum(nurse_vars[h, d] for h in range(n_shifts)) == 1, f"Department_{d}_Scheduling")

        for h in range(n_shifts):
            model.addCons(quicksum(medical_equipment[d] * nurse_vars[h, d] for d in range(n_departments)) <= room_capacities[h], f"Shift_{h}_Capacity")

        for edge in graph.edges:
            model.addCons(doctor_vars[edge[0]] + doctor_vars[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_shifts):
            model.addCons(
                quicksum(equipment_usage_vars[i, j] for j in range(n_shifts) if i != j) ==
                quicksum(equipment_usage_vars[j, i] for j in range(n_shifts) if i != j),
                f"Equipment_Usage_Conservation_{i}"
            )

        for j in range(n_shifts):
            model.addCons(
                quicksum(equipment_transport_vars[i, j] for i in range(n_shifts) if i != j) ==
                quicksum(nurse_vars[j, d] for d in range(n_departments)),
                f"Equipment_Transport_Conservation_{j}"
            )

        for i in range(n_shifts):
            for j in range(n_shifts):
                if i != j:
                    model.addCons(equipment_transport_vars[i, j] <= transportation_capacities[i, j], f"Equipment_Transport_Capacity_{i}_{j}")

        for h in range(n_shifts):
            model.addCons(
                quicksum(nurse_vars[h, d] for d in range(n_departments)) <= n_departments * doctor_vars[h],
                f"Equipment_Scheduling_{h}"
            )

        for i in range(n_shifts):
            model.addCons(
                quicksum(equipment_transport_vars[i, j] for j in range(n_shifts)) <= (self.department_capacity * department_vars[i]),
                f"Department_Capacity_{i}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_shifts': 33,
        'n_departments': 63,
        'min_doctor_cost': 3000,
        'max_doctor_cost': 3000,
        'min_nurse_cost': 600,
        'max_nurse_cost': 625,
        'min_room_capacity': 37,
        'max_room_capacity': 150,
        'link_probability': 0.17,
        'transportation_capacity_min': 37,
        'transportation_capacity_max': 2500,
        'hiring_penalty_min': 0.24,
        'hiring_penalty_max': 0.5,
        'department_capacity': 1000,
    }

    resource_manager = HospitalResourceManagement(parameters, seed=seed)
    instance = resource_manager.generate_instance()
    solve_status, solve_time, objective_value = resource_manager.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")