import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CityNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data generation #################
    def generate_instance(self):
        assert self.min_cost_per_km >= 0 and self.max_cost_per_km >= self.min_cost_per_km
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand

        # Create a random graph for city network
        G = nx.erdos_renyi_graph(self.n_nodes, self.prob_edge)

        Network_Cost = np.zeros((self.n_nodes, self.n_nodes))
        for (u, v) in G.edges():
            cost = random.uniform(self.min_cost_per_km, self.max_cost_per_km)
            Network_Cost[u, v] = cost
            Network_Cost[v, u] = cost

        Home_Device = np.random.randint(1, self.max_capacity, size=self.n_nodes)
        Customer_Demand = np.zeros(self.n_nodes)
        for i in range(self.n_customers):
            node = random.randint(0, self.n_nodes - 1)
            demand = random.uniform(self.min_demand, self.max_demand)
            Customer_Demand[node] += demand

        return {
            "Network_Cost": Network_Cost,
            "Home_Device": Home_Device,
            "Customer_Demand": Customer_Demand
        }
    
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        Network_Cost = instance['Network_Cost']
        Home_Device = instance['Home_Device']
        Customer_Demand = instance['Customer_Demand']
        
        model = Model("CityNetworkOptimization")

        n = Network_Cost.shape[0]
        
        # Decision variables
        edge_vars = {(u, v): model.addVar(vtype="B", name=f"Edge_{u}_{v}") for u in range(n) for v in range(n) if Network_Cost[u, v] > 0}
        flow_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}", lb=0) for u in range(n) for v in range(n) if Network_Cost[u, v] > 0}
        
        # Objective: minimize total network cost
        objective_expr = quicksum(edge_vars[u, v] * Network_Cost[u, v] for u in range(n) for v in range(n) if Network_Cost[u, v] > 0)
        
        # Constraints: Capacity constraints per home device
        for node in range(n):
            in_flow = quicksum(flow_vars[u, node] for u in range(n) if (u, node) in flow_vars)
            out_flow = quicksum(flow_vars[node, v] for v in range(n) if (node, v) in flow_vars)
            model.addCons(in_flow - out_flow <= Home_Device[node], f"Capacity_{node}")

        # Constraints: Flow must meet customer demand
        for node in range(n):
            model.addCons(quicksum(flow_vars[node, v] for v in range(n) if (node, v) in flow_vars) >= Customer_Demand[node], f"Demand_{node}")
        
        # Constraints: Flow on an edge can only be positive if the edge is selected
        for u in range(n):
            for v in range(n):
                if (u, v) in flow_vars:
                    model.addCons(flow_vars[u, v] <= edge_vars[u, v] * Home_Device[u], f"Capacity_flow_{u}_{v}")

        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "minimize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 150,
        'n_customers': 60,
        'min_cost_per_km': 300,
        'max_cost_per_km': 2000,
        'max_capacity': 700,
        'min_demand': 100,
        'max_demand': 200,
        'prob_edge': 0.59,
    }
    ### new parameter code ends here

    network_optimization = CityNetworkOptimization(parameters, seed=seed)
    instance = network_optimization.generate_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")