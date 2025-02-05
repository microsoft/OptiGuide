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

class CityHealthcareResourceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def get_instance(self):
        personnel_costs = np.random.randint(self.min_personnel_cost, self.max_personnel_cost + 1, self.n_hospitals)
        vaccine_costs = np.random.randint(self.min_vaccine_cost, self.max_vaccine_cost + 1, (self.n_hospitals, self.n_vaccines))
        capacities = np.random.randint(self.min_hospital_capacity, self.max_hospital_capacity + 1, self.n_hospitals)
        vaccines = np.random.gamma(2, 2, self.n_vaccines).astype(int) + 1

        graph = Graph.erdos_renyi(self.n_hospitals, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        transport_costs = np.random.randint(1, 20, (self.n_hospitals, self.n_hospitals))
        transport_capacities = np.random.randint(self.transport_capacity_min, self.transport_capacity_max + 1, (self.n_hospitals, self.n_hospitals))

        environmental_impact = np.random.uniform(0.1, 10, (self.n_hospitals, self.n_hospitals))
        penalty_costs = np.random.uniform(10, 50, self.n_hospitals)
        
        return {
            "personnel_costs": personnel_costs,
            "vaccine_costs": vaccine_costs,
            "capacities": capacities,
            "vaccines": vaccines,
            "graph": graph,
            "edge_weights": edge_weights,
            "transport_costs": transport_costs,
            "transport_capacities": transport_capacities,
            "environmental_impact": environmental_impact,
            "penalty_costs": penalty_costs
        }

    def solve(self, instance):
        personnel_costs = instance['personnel_costs']
        vaccine_costs = instance['vaccine_costs']
        capacities = instance['capacities']
        vaccines = instance['vaccines']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        transport_costs = instance['transport_costs']
        transport_capacities = instance['transport_capacities']
        environmental_impact = instance['environmental_impact']
        penalty_costs = instance['penalty_costs']

        model = Model("CityHealthcareResourceOptimization")
        n_hospitals = len(personnel_costs)
        n_vaccines = len(vaccine_costs[0])

        neighborhood_alloc = {h: model.addVar(vtype="B", name=f"NeighborhoodAllocation_{h}") for h in range(n_hospitals)}
        community_syringes = {(h, v): model.addVar(vtype="B", name=f"Hospital_{h}_Vaccine_{v}") for h in range(n_hospitals) for v in range(n_vaccines)}
        health_supplies = {(i, j): model.addVar(vtype="I", name=f"HealthSupplies_{i}_{j}") for i in range(n_hospitals) for j in range(n_hospitals)}
        city_transport = {(i, j): model.addVar(vtype="I", name=f"CityTransport_{i}_{j}") for i in range(n_hospitals) for j in range(n_hospitals)}

        model.setObjective(
            quicksum(personnel_costs[h] * neighborhood_alloc[h] for h in range(n_hospitals)) +
            quicksum(vaccine_costs[h, v] * community_syringes[h, v] for h in range(n_hospitals) for v in range(n_vaccines)) +
            quicksum(transport_costs[i, j] * city_transport[i, j] for i in range(n_hospitals) for j in range(n_hospitals)) +
            quicksum(environmental_impact[i, j] * city_transport[i, j] for i in range(n_hospitals) for j in range(n_hospitals)) +
            quicksum(penalty_costs[i] for i in range(n_hospitals)),
            "minimize"
        )

        for v in range(n_vaccines):
            model.addCons(quicksum(community_syringes[h, v] for h in range(n_hospitals)) == 1, f"Vaccine_{v}_Allocation")
    
        for h in range(n_hospitals):
            for v in range(n_vaccines):
                model.addCons(community_syringes[h, v] <= neighborhood_alloc[h], f"Hospital_{h}_Serve_{v}")
    
        for h in range(n_hospitals):
            model.addCons(quicksum(vaccines[v] * community_syringes[h, v] for v in range(n_vaccines)) <= capacities[h], f"Hospital_{h}_Capacity")

        for edge in graph.edges:
            model.addCons(neighborhood_alloc[edge[0]] + neighborhood_alloc[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_hospitals):
            model.addCons(
                quicksum(health_supplies[i, j] for j in range(n_hospitals) if i != j) ==
                quicksum(health_supplies[j, i] for j in range(n_hospitals) if i != j),
                f"Supply_Conservation_{i}"
            )
    
        for j in range(n_hospitals):
            model.addCons(
                quicksum(city_transport[i, j] for i in range(n_hospitals) if i != j) ==
                quicksum(community_syringes[j, v] for v in range(n_vaccines)),
                f"Transport_Conservation_{j}"
            )
    
        for i in range(n_hospitals):
            for j in range(n_hospitals):
                if i != j:
                    model.addCons(city_transport[i, j] <= transport_capacities[i, j], f"Transport_Capacity_{i}_{j}")
    
        for h in range(n_hospitals):
            model.addCons(
                quicksum(community_syringes[h, v] for v in range(n_vaccines)) <= n_vaccines * neighborhood_alloc[h],
                f"Convex_Hull_{h}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hospitals': 24,
        'n_vaccines': 252,
        'min_vaccine_cost': 619,
        'max_vaccine_cost': 3000,
        'min_personnel_cost': 1397,
        'max_personnel_cost': 5000,
        'min_hospital_capacity': 295,
        'max_hospital_capacity': 2154,
        'link_probability': 0.17,
        'transport_capacity_min': 1170,
        'transport_capacity_max': 3000,
    }

    resource_optimizer = CityHealthcareResourceOptimization(parameters, seed=seed)
    instance = resource_optimizer.get_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")