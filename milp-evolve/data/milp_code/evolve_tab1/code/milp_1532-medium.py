import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexManufacturingOptimization:
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
    
    def generate_energy_costs(self):
        return np.random.uniform(self.energy_cost_interval[0], self.energy_cost_interval[1], (self.n_customers, self.n_facilities))
    
    def generate_production_stages(self):
        return self.randint(self.n_products, self.production_stage_interval)
    
    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.fixed_cost * np.ones(self.n_facilities)
        production_stages = self.generate_production_stages()
        raw_material_costs = self.randint(self.n_products, self.raw_material_cost_interval)

        # Generate a random graph to simulate the network
        G = nx.erdos_renyi_graph(self.n_customers + self.n_facilities, self.connection_density, directed=True)
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.uniform(*self.time_interval)
        
        delivery_times = self.generate_delivery_times(G)

        energy_costs = self.generate_energy_costs()

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'delivery_times': delivery_times,
            'energy_costs': energy_costs,
            'production_stages': production_stages,
            'raw_material_costs': raw_material_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        delivery_times = instance['delivery_times']
        energy_costs = instance['energy_costs']
        production_stages = instance['production_stages']
        raw_material_costs = instance['raw_material_costs']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        n_products = len(production_stages)
        
        model = Model("ComplexManufacturingOptimization")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        allocation = {(i, j): model.addVar(vtype="B", name=f"Alloc_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        production_vars = {(p, s): model.addVar(vtype="B", name=f"Prod_{p}_{s}") for p in range(n_products) for s in range(production_stages[p])}

        # Objective: Minimize the total cost including delivery time cost, fixed facility opening cost, and energy consumption cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum(delivery_times[i, j] * allocation[i, j] for i in range(n_customers) for j in range(n_facilities)) + \
                         quicksum(energy_costs[i, j] * allocation[i, j] for i in range(n_customers) for j in range(n_facilities))
                         
        # Constraints: Ensure each customer demand is fully met
        for i in range(n_customers):
            model.addCons(quicksum(allocation[i, j] for j in range(n_facilities)) == 1, f"DemandMet_{i}")
        
        # Constraints: Ensure facility capacity is not exceeded
        for j in range(n_facilities):
            capacity_expr = quicksum(allocation[i, j] * demands[i] for i in range(n_customers))
            model.addCons(capacity_expr <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        # Constraints: Ensure production stages follow precedence constraints
        for p in range(n_products):
            for s in range(production_stages[p] - 1):
                model.addCons(production_vars[p, s] <= production_vars[p, s+1], f"Precedence_{p}_{s}")
        
        # Constraints: Add raw material cost constraints
        for p in range(n_products):
            total_cost = quicksum(raw_material_costs[p] * production_vars[p, s] for s in range(production_stages[p]))
            model.addCons(total_cost <= self.max_raw_material_cost, f"RawMaterialCost_{p}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 60,
        'n_facilities': 225,
        'n_products': 90,
        'demand_interval': (360, 1440),
        'capacity_interval': (300, 800),
        'fixed_cost': 750,
        'time_interval': (160, 960),
        'energy_cost_interval': (700, 2100),
        'production_stage_interval': (1, 4),
        'raw_material_cost_interval': (25, 125),
        'connection_density': 0.38,
        'ratio': 100,
        'max_raw_material_cost': 2000,
    }

    complex_opt = ComplexManufacturingOptimization(parameters, seed=seed)
    instance = complex_opt.generate_instance()
    solve_status, solve_time = complex_opt.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")