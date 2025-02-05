import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class LocationRoutingProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_network(self):
        # Erdos-Renyi graph to simulate customer-hub potential connections
        G = nx.erdos_renyi_graph(n=self.n_locations, p=self.connection_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_locations, self.n_locations), dtype=object)
        hub_list = range(self.n_locations)
        customer_list = range(self.n_locations)
        connections = []

        for i, j in G.edges:
            cost_ij = np.random.uniform(*self.cost_range)
            capacity_ij = np.random.uniform(self.capacity_min, self.capacity_max)
            adj_mat[i, j] = (cost_ij, capacity_ij)
            connections.append((i, j))

        return G, adj_mat, connections, hub_list, customer_list

    def generate_customer_demand(self, customer_list):
        demands = np.random.randint(self.demand_min, self.demand_max, size=len(customer_list))
        return demands
    
    def generate_hub_capacity(self, hub_list):
        capacities = np.random.randint(self.capacity_min, self.capacity_max, size=len(hub_list))
        return capacities

    def generate_instance(self):
        G, adj_mat, connections, hub_list, customer_list = self.generate_network()
        customer_demands = self.generate_customer_demand(customer_list)
        hub_capacities = self.generate_hub_capacity(hub_list)

        res = {
            'adj_mat': adj_mat, 
            'connections': connections, 
            'customer_demands': customer_demands, 
            'hub_capacities': hub_capacities, 
            'hub_list': hub_list,
            'customer_list': customer_list
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        connections = instance['connections']
        customer_demands = instance['customer_demands']
        hub_capacities = instance['hub_capacities']
        hub_list = instance['hub_list']
        customer_list = instance['customer_list']
        
        model = Model("LocationRoutingProblem")

        # Variables
        routeflow_vars = {f"routeflow_{h}_{c}": model.addVar(vtype="I", name=f"routeflow_{h}_{c}") for h in hub_list for c in customer_list if (h, c) in connections}
        hubopen_vars = {f"hubopen_{h}": model.addVar(vtype="B", name=f"hubopen_{h}") for h in hub_list}
        
        # Objective Function
        operating_cost = quicksum(
            adj_mat[h, c][0] * routeflow_vars[f"routeflow_{h}_{c}"] for h in hub_list for c in customer_list if (h, c) in connections
        )
        hub_cost = quicksum(
            hubopen_vars[f"hubopen_{h}"] * self.fixed_hub_open_cost for h in hub_list
        )
        
        model.setObjective(operating_cost + hub_cost, "minimize")

        # Constraints
        for c in customer_list:
            model.addCons(
                quicksum(routeflow_vars[f"routeflow_{h}_{c}"] for h in hub_list if (h, c) in connections) == customer_demands[c],
                f"costroute_{c}"
            )

        for h in hub_list:
            model.addCons(
                quicksum(routeflow_vars[f"routeflow_{h}_{c}"] for c in customer_list if (h, c) in connections) <= hub_capacities[h] * hubopen_vars[f"hubopen_{h}"],
                f"hubcapacity_{h}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 175,
        'connection_prob': 0.53,
        'cost_range': (30, 120),
        'demand_min': 80,
        'demand_max': 250,
        'capacity_min': 700,
        'capacity_max': 2700,
        'fixed_hub_open_cost': 1400,
    }

    lrp = LocationRoutingProblem(parameters, seed=seed)
    instance = lrp.generate_instance()
    solve_status, solve_time = lrp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")