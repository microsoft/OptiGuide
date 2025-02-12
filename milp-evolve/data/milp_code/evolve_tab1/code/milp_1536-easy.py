import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def generate_delivery_times(self, graph):
        times = nx.get_edge_attributes(graph, 'weight')
        delivery_times = np.zeros((self.n_customers, self.n_facilities))
        
        for i in range(self.n_customers):
            for j in range(self.n_facilities):
                time = times.get((i, j), random.uniform(*self.time_interval))
                delivery_times[i, j] = time + np.random.normal(self.delivery_mean, self.delivery_std_dev)
        
        return delivery_times
        
    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.fixed_cost * np.ones(self.n_facilities)

        # Generate a random graph to simulate the network
        G = nx.erdos_renyi_graph(self.n_customers + self.n_facilities, self.connection_density, directed=True)
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.uniform(*self.time_interval)
        
        delivery_times = self.generate_delivery_times(G)

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'delivery_times': delivery_times,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        delivery_times = instance['delivery_times']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("SimplifiedLogisticsOptimization")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        allocation = {(i, j): model.addVar(vtype="B", name=f"Alloc_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Objective: Minimize the total cost including delivery time cost and fixed facility opening cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum(delivery_times[i, j] * allocation[i, j] for i in range(n_customers) for j in range(n_facilities))
                         
        # Constraints: Ensure each customer demand is fully met
        for i in range(n_customers):
            model.addCons(quicksum(allocation[i, j] for j in range(n_facilities)) == 1, f"DemandMet_{i}")
        
        # Constraints: Ensure facility capacity is not exceeded
        for j in range(n_facilities):
            capacity_expr = quicksum(allocation[i, j] * demands[i] for i in range(n_customers))
            model.addCons(capacity_expr <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        # New constraints for Convex Hull Formulation
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(allocation[i, j] <= open_facilities[j], f"Link_{i}_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 30,
        'n_facilities': 300,
        'demand_interval': (120, 480),
        'capacity_interval': (900, 2400),
        'fixed_cost': 1500,
        'time_interval': (320, 1920),
        'connection_density': 0.73,
        'ratio': 400,
        'delivery_mean': 800,
        'delivery_std_dev': 150,
    }

    logistics_optimization = SimplifiedLogisticsOptimization(parameters, seed=seed)
    instance = logistics_optimization.generate_instance()
    solve_status, solve_time = logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")