import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexLogisticsOptimization:
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
                delivery_times[i, j] = times.get((i, j), random.uniform(*self.time_interval))
        
        return delivery_times

    def generate_random_uncertainties(self, shape, std):
        return np.random.normal(0, std, shape)
        
    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.fixed_cost * np.ones(self.n_facilities)

        # Generate a random graph to simulate the network
        G = nx.erdos_renyi_graph(self.n_customers + self.n_facilities, self.connection_density, directed=True)
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.uniform(*self.time_interval)

        delivery_times = self.generate_delivery_times(G)
        uncertainties = self.generate_random_uncertainties(delivery_times.shape, self.epsilon_std)

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'delivery_times': delivery_times,
            'uncertainties': uncertainties
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        delivery_times = instance['delivery_times']
        uncertainties = instance['uncertainties']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("ComplexLogisticsOptimization")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        allocation = {(i, j): model.addVar(vtype="B", name=f"Alloc_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Objective: Minimize the total cost including delivery time cost and fixed facility opening cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum((delivery_times[i, j] + uncertainties[i, j]) * allocation[i, j] for i in range(n_customers) for j in range(n_facilities))
                         
        # Constraints: Ensure each customer demand is fully met
        for i in range(n_customers):
            model.addCons(quicksum(allocation[i, j] for j in range(n_facilities)) == 1, f"DemandMet_{i}")
        
        # Constraints: Ensure facility capacity is not exceeded
        for j in range(n_facilities):
            capacity_expr = quicksum(allocation[i, j] * demands[i] for i in range(n_customers))
            model.addCons(capacity_expr <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 20,
        'n_facilities': 200,
        'demand_interval': (10, 100),
        'capacity_interval': (600, 1500),
        'fixed_cost': 1125,
        'time_interval': (35, 350),
        'connection_density': 0.66,
        'ratio': 56.25,
        'epsilon_std': 1.25,
    }

    logistics_optimization = ComplexLogisticsOptimization(parameters, seed=seed)
    instance = logistics_optimization.generate_instance()
    solve_status, solve_time = logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")