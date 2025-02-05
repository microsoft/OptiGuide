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

class FactoryOperationManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        manager_costs = np.random.randint(self.min_manager_cost, self.max_manager_cost + 1, self.n_shifts)
        security_costs = np.random.randint(self.min_security_cost, self.max_security_cost + 1, (self.n_shifts, self.n_zones))
        machine_capacities = np.random.randint(self.min_machine_capacity, self.max_machine_capacity + 1, self.n_shifts)
        equipment_repair_times = np.random.gamma(2, 2, self.n_zones).astype(int) + 1

        graph = Graph.erdos_renyi(self.n_shifts, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        transportation_costs = np.random.randint(1, 20, (self.n_shifts, self.n_shifts))
        transportation_capacities = np.random.randint(self.transportation_capacity_min, self.transportation_capacity_max + 1, (self.n_shifts, self.n_shifts))

        hazard_transport_costs = np.random.randint(10, 50, (self.n_shifts, self.n_shifts))
        maintenance_costs = np.random.randint(5, 30, (self.n_shifts, self.n_shifts))
        cyber_security_penalties = np.random.uniform(low=0.5, high=2.0, size=(self.n_shifts, self.n_shifts))

        compliance_costs = np.random.randint(10, 1000, self.n_zones)
        
        factory_risk = np.random.uniform(0.1, 10, (self.n_shifts, self.n_shifts))

        return {
            "manager_costs": manager_costs,
            "security_costs": security_costs,
            "machine_capacities": machine_capacities,
            "equipment_repair_times": equipment_repair_times,
            "graph": graph,
            "edge_weights": edge_weights,
            "transportation_costs": transportation_costs,
            "transportation_capacities": transportation_capacities,
            "hazard_transport_costs": hazard_transport_costs,
            "maintenance_costs": maintenance_costs,
            "cyber_security_penalties": cyber_security_penalties,
            "compliance_costs": compliance_costs,
            "factory_risk": factory_risk,
        }

    def solve(self, instance):
        manager_costs = instance['manager_costs']
        security_costs = instance['security_costs']
        machine_capacities = instance['machine_capacities']
        equipment_repair_times = instance['equipment_repair_times']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        transportation_costs = instance['transportation_costs']
        transportation_capacities = instance['transportation_capacities']
        hazard_transport_costs = instance['hazard_transport_costs']
        maintenance_costs = instance['maintenance_costs']
        cyber_security_penalties = instance['cyber_security_penalties']
        compliance_costs = instance['compliance_costs']
        factory_risk = instance['factory_risk']

        model = Model("FactoryOperationManagement")
        n_shifts = len(manager_costs)
        n_zones = len(security_costs[0])

        manager_vars = {h: model.addVar(vtype="B", name=f"Manager_Scheduling_{h}") for h in range(n_shifts)}
        security_vars = {(h, z): model.addVar(vtype="B", name=f"Shift_{h}_Zone_{z}") for h in range(n_shifts) for z in range(n_zones)}
        equipment_maintenance_vars = {(i, j): model.addVar(vtype="I", name=f"Maintenance_Time_{i}_{j}") for i in range(n_shifts) for j in range(n_shifts)}
        hazardous_transport_vars = {(i, j): model.addVar(vtype="I", name=f"Hazard_Transport_{i}_{j}") for i in range(n_shifts) for j in range(n_shifts)}

        machine_usage_vars = {(i, j): model.addVar(vtype="C", name=f"Machine_Usage_{i}_{j}") for i in range(n_shifts) for j in range(n_shifts)}

        cyber_security_vars = {i: model.addVar(vtype="B", name=f"Cyber_Security_{i}") for i in range(n_shifts)}

        model.setObjective(
            quicksum(manager_costs[h] * manager_vars[h] for h in range(n_shifts)) +
            quicksum(security_costs[h, z] * security_vars[h, z] for h in range(n_shifts) for z in range(n_zones)) +
            quicksum(transportation_costs[i, j] * hazardous_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(hazard_transport_costs[i, j] * hazardous_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(maintenance_costs[i, j] * hazardous_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(cyber_security_penalties[i, j] * hazardous_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)) +
            quicksum(compliance_costs[z] * security_vars[h, z] for h in range(n_shifts) for z in range(n_zones)) +
            quicksum(factory_risk[i, j] * hazardous_transport_vars[i, j] for i in range(n_shifts) for j in range(n_shifts)),
            "minimize"
        )

        for z in range(n_zones):
            model.addCons(quicksum(security_vars[h, z] for h in range(n_shifts)) == 1, f"Zone_{z}_Security")

        for h in range(n_shifts):
            model.addCons(quicksum(equipment_repair_times[z] * security_vars[h, z] for z in range(n_zones)) <= machine_capacities[h], f"Shift_{h}_Capacity")

        for edge in graph.edges:
            model.addCons(manager_vars[edge[0]] + manager_vars[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_shifts):
            model.addCons(
                quicksum(equipment_maintenance_vars[i, j] for j in range(n_shifts) if i != j) ==
                quicksum(equipment_maintenance_vars[j, i] for j in range(n_shifts) if i != j),
                f"Equipment_Maintenance_Conservation_{i}"
            )

        for j in range(n_shifts):
            model.addCons(
                quicksum(hazardous_transport_vars[i, j] for i in range(n_shifts) if i != j) ==
                quicksum(security_vars[j, z] for z in range(n_zones)),
                f"Equipment_Transport_Conservation_{j}"
            )

        for i in range(n_shifts):
            for j in range(n_shifts):
                if i != j:
                    model.addCons(hazardous_transport_vars[i, j] <= transportation_capacities[i, j], f"Hazard_Transport_Capacity_{i}_{j}")

        for h in range(n_shifts):
            model.addCons(
                quicksum(security_vars[h, z] for z in range(n_zones)) <= n_zones * manager_vars[h],
                f"Equipment_Scheduling_{h}"
            )

        for i in range(n_shifts):
            model.addCons(
                quicksum(hazardous_transport_vars[i, j] for j in range(n_shifts)) <= (self.zone_capacity * cyber_security_vars[i]),
                f"Zone_Capacity_{i}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_shifts': 33,
        'n_zones': 126,
        'min_manager_cost': 3000,
        'max_manager_cost': 3000,
        'min_security_cost': 600,
        'max_security_cost': 1875,
        'min_machine_capacity': 37,
        'max_machine_capacity': 450,
        'link_probability': 0.45,
        'transportation_capacity_min': 333,
        'transportation_capacity_max': 1875,
        'maintenance_penalty_min': 0.73,
        'maintenance_penalty_max': 0.73,
        'zone_capacity': 750,
    }

    factory_manager = FactoryOperationManagement(parameters, seed=seed)
    instance = factory_manager.generate_instance()
    solve_status, solve_time, objective_value = factory_manager.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")