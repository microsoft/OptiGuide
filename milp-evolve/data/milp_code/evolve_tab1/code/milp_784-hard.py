import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class RetailNetworkDesignOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_hubs = random.randint(self.min_hubs, self.max_hubs)
        num_stores = random.randint(self.min_stores, self.max_stores)

        # Cost matrices
        transport_cost = np.random.randint(50, 300, size=(num_stores, num_hubs))
        opening_costs_hub = np.random.randint(1000, 3000, size=num_hubs)

        # Store demands
        store_demands = np.random.randint(100, 500, size=num_stores)

        # Hub capacities
        hub_capacity = np.random.randint(5000, 10000, size=num_hubs)

        # Distance matrices
        max_acceptable_distance = np.random.randint(50, 200)
        store_distances_from_hubs = np.random.randint(10, 500, size=(num_stores, num_hubs))

        res = {
            'num_hubs': num_hubs,
            'num_stores': num_stores,
            'transport_cost': transport_cost,
            'opening_costs_hub': opening_costs_hub,
            'store_demands': store_demands,
            'hub_capacity': hub_capacity,
            'store_distances_from_hubs': store_distances_from_hubs,
            'max_acceptable_distance': max_acceptable_distance
        }
        
        # Generate road network graph
        G = nx.erdos_renyi_graph(num_stores + num_hubs, self.road_connectivity_prob, seed=self.seed)
        res['road_network'] = list(G.edges)

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_hubs = instance['num_hubs']
        num_stores = instance['num_stores']
        transport_cost = instance['transport_cost']
        opening_costs_hub = instance['opening_costs_hub']
        store_demands = instance['store_demands']
        hub_capacity = instance['hub_capacity']
        store_distances_from_hubs = instance['store_distances_from_hubs']
        max_acceptable_distance = instance['max_acceptable_distance']
        road_network = instance['road_network']
        
        model = Model("RetailNetworkDesignOptimization")

        # Variables
        Number_of_Hubs = {j: model.addVar(vtype="B", name=f"Number_of_Hubs_{j}") for j in range(num_hubs)}
        Customer_Satisfaction = {(i, j): model.addVar(vtype="B", name=f"Customer_Satisfaction_{i}_{j}") for i in range(num_stores) for j in range(num_hubs)}
        Road_Opened = {(u, v): model.addVar(vtype="B", name=f"Road_Opened_{u}_{v}") for u, v in road_network}

        # Objective function: Minimize total costs
        total_cost = quicksum(Customer_Satisfaction[i, j] * transport_cost[i, j] for i in range(num_stores) for j in range(num_hubs)) + \
                     quicksum(Number_of_Hubs[j] * opening_costs_hub[j] for j in range(num_hubs))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_stores):
            model.addCons(quicksum(Customer_Satisfaction[i, j] for j in range(num_hubs)) == 1, name=f"store_served_{i}")

        for j in range(num_hubs):
            for i in range(num_stores):
                model.addCons(Customer_Satisfaction[i, j] <= Number_of_Hubs[j], name=f"hub_open_{i}_{j}")

        for j in range(num_hubs):
            model.addCons(quicksum(store_demands[i] * Customer_Satisfaction[i, j] for i in range(num_stores)) <= hub_capacity[j], name=f"hub_capacity_{j}")
        
        for i in range(num_stores):
            for j in range(num_hubs):
                model.addCons(store_distances_from_hubs[i, j] * Customer_Satisfaction[i, j] <= max_acceptable_distance, name=f"distance_constraint_{i}_{j}")

        min_service_level = 5  # Example minimum service level; can be a parameter
        for j in range(num_hubs):
            model.addCons(quicksum(Customer_Satisfaction[i, j] for i in range(num_stores)) >= min_service_level * Number_of_Hubs[j], name=f"min_service_level_{j}")

        # Incorporate road network constraints
        for u, v in road_network:
            store_hub_assignment_u = quicksum(Customer_Satisfaction[u, j] for j in range(num_hubs) if u < num_stores)
            store_hub_assignment_v = quicksum(Customer_Satisfaction[v, j] for j in range(num_hubs) if v < num_stores)
            model.addCons(store_hub_assignment_u + store_hub_assignment_v <= Road_Opened[u, v] + 1, name=f"road_network_{u}_{v}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_hubs': 375,
        'max_hubs': 675,
        'min_stores': 50,
        'max_stores': 200,
        'road_connectivity_prob': 0.45,
    }

    optimization = RetailNetworkDesignOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")