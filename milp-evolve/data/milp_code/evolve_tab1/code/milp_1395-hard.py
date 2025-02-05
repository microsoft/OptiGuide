import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Network:
    def __init__(self, number_of_hubs, links, reliabilities, neighbors):
        self.number_of_hubs = number_of_hubs
        self.hubs = np.arange(number_of_hubs)
        self.links = links
        self.reliabilities = reliabilities
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_hubs, probability):
        edges = [(i, j) for i in range(number_of_hubs) for j in range(i+1, number_of_hubs) if np.random.rand() < probability]
        reliabilities = np.random.rand(number_of_hubs)
        neighbors = {hub: set() for hub in range(number_of_hubs)}
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)

        return Network(number_of_hubs, edges, reliabilities, neighbors)

class TelecommunicationHubNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.num_hubs > 0 and self.num_customers > 0
        assert self.min_hub_cost >= 0 and self.max_hub_cost >= self.min_hub_cost
        assert self.min_service_cost >= 0 and self.max_service_cost >= self.min_service_cost
        assert self.min_hub_capacity > 0 and self.max_hub_capacity >= self.min_hub_capacity

        hub_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.num_hubs)
        service_costs = np.random.randint(self.min_service_cost, self.max_service_cost + 1, (self.num_hubs, self.num_customers))
        capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.num_hubs)
        demands = np.random.randint(1, 20, self.num_customers)

        service_times = np.random.randint(1, 10, (self.num_hubs, self.num_customers))
        maintenance_costs = np.random.randint(100, 1000, self.num_hubs)
        hub_reliabilities = np.random.rand(self.num_hubs)

        network = Network.erdos_renyi(self.num_hubs, self.connectivity_probability)
        cliques = []
        for clique in nx.find_cliques(nx.Graph(network.links)):
            if len(clique) > 1:
                cliques.append(tuple(sorted(clique)))

        return {
            "hub_costs": hub_costs,
            "service_costs": service_costs,
            "capacities": capacities,
            "demands": demands,
            "cliques": cliques,
            "service_times": service_times,
            "maintenance_costs": maintenance_costs,
            "hub_reliabilities": hub_reliabilities
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hub_costs = instance["hub_costs"]
        service_costs = instance["service_costs"]
        capacities = instance["capacities"]
        demands = instance["demands"]
        cliques = instance["cliques"]
        service_times = instance["service_times"]
        maintenance_costs = instance["maintenance_costs"]
        hub_reliabilities = instance["hub_reliabilities"]
        
        model = Model("TelecommunicationHubNetworkDesign")
        num_hubs = len(hub_costs)
        num_customers = len(service_costs[0])
        
        # Decision variables
        hub_selection = {h: model.addVar(vtype="B", name=f"HubSelection_{h}") for h in range(num_hubs)}
        customer_assignment = {(h, c): model.addVar(vtype="B", name=f"Hub_{h}_Customer_{c}") for h in range(num_hubs) for c in range(num_customers)}
        service_time_vars = {(h, c): model.addVar(vtype="C", name=f"ServiceTime_{h}_{c}") for h in range(num_hubs) for c in range(num_customers)}
        reliability_vars = {h: model.addVar(vtype="C", name=f"Reliability_{h}") for h in range(num_hubs)}

        # Objective: minimize total hub setup, service costs, and maintenance costs
        model.setObjective(
            quicksum(hub_costs[h] * hub_selection[h] for h in range(num_hubs)) +
            quicksum(service_costs[h, c] * customer_assignment[h, c] for h in range(num_hubs) for c in range(num_customers)) +
            quicksum(maintenance_costs[h] * hub_selection[h] for h in range(num_hubs)), "minimize"
        )
        
        # Constraints: Each customer is assigned to at least one hub
        for c in range(num_customers):
            model.addCons(quicksum(customer_assignment[h, c] for h in range(num_hubs)) >= 1, f"Customer_{c}_Coverage")
        
        # Constraints: Only selected hubs can serve customers
        for h in range(num_hubs):
            for c in range(num_customers):
                model.addCons(customer_assignment[h, c] <= hub_selection[h], f"Hub_{h}_Service_{c}")
                model.addCons(service_time_vars[h, c] >= service_times[h, c] * customer_assignment[h, c], f"ServiceTime_{h}_{c}")

        # Constraints: Hubs cannot exceed their capacity
        for h in range(num_hubs):
            model.addCons(quicksum(demands[c] * customer_assignment[h, c] for c in range(num_customers)) <= capacities[h], f"Hub_{h}_CapacityLimit")
        
        # Constraints: Hub Clique Limits
        for count, clique in enumerate(cliques):
            model.addCons(quicksum(hub_selection[node] for node in clique) <= 1, f"HubCliqueRestriction_{count}")

        # Constraints: Hub reliability
        for h in range(num_hubs):
            model.addCons(reliability_vars[h] == hub_reliabilities[h], f"HubReliability_{h}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_hubs': 50,
        'num_customers': 100,
        'min_service_cost': 90,
        'max_service_cost': 1200,
        'min_hub_cost': 2000,
        'max_hub_cost': 2500,
        'min_hub_capacity': 300,
        'max_hub_capacity': 1000,
        'connectivity_probability': 0.31,
    }
    
    hub_optimizer = TelecommunicationHubNetworkDesign(parameters, seed)
    instance = hub_optimizer.generate_instance()
    solve_status, solve_time, objective_value = hub_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")