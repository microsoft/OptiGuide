import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GISP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['cost'] = 1.0  # Simplified uniform cost

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_zone_data(self, G):
        return {node: np.random.randint(self.min_zone_limit, self.max_zone_limit) for node in G.nodes}
    
    def generate_labor_data(self):
        labor_costs = {f"l{node}": np.random.randint(self.min_labor_cost, self.max_labor_cost) for node in range(self.num_labor_resources)}
        labor_quotas = {f"l{node}": np.random.randint(self.min_quota, self.max_quota) for node in range(self.num_labor_resources)}
        max_working_hours = {f"l{node}": np.random.randint(self.min_working_hours, self.max_working_hours) for node in range(self.num_labor_resources)}
        return labor_costs, labor_quotas, max_working_hours

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        zone_limits = self.generate_zone_data(G)
        
        res = {'G': G, 
               'E2': E2,
               'zone_limits': zone_limits}
        
        res['node_capacities'] = {node: np.random.randint(self.min_capacity, self.max_capacity) for node in G.nodes}
        res['max_removable_edges'] = np.random.randint(self.min_removable_edges, self.max_removable_edges)

        labor_costs, labor_quotas, max_working_hours = self.generate_labor_data()
        res['labor_costs'] = labor_costs
        res['labor_quotas'] = labor_quotas
        res['max_working_hours'] = max_working_hours

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        node_capacities = instance['node_capacities']
        max_removable_edges = instance['max_removable_edges']
        zone_limits = instance['zone_limits']
        labor_costs = instance['labor_costs']
        labor_quotas = instance['labor_quotas']
        max_working_hours = instance['max_working_hours']

        model = Model("GISP")

        # Define Variables
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        production_yield = {f"Y{node}": model.addVar(vtype="C", name=f"Y{node}", lb=0) for node in G.nodes}
        U_labor = {f"U_l{node}": model.addVar(vtype="C", name=f"U_l{node}", lb=0) for node in range(self.num_labor_resources)}
        bal_factor = {f"B_l{node}": model.addVar(vtype="C", name=f"B_l{node}", lb=0) for node in range(self.num_labor_resources)}
        
        # Objective: Maximize revenues and minimize costs while maximizing labor utilization
        objective_expr = (quicksum(G.nodes[node]['revenue'] * node_vars[f"x{node}"] for node in G.nodes) - 
                          quicksum(G[u][v]['cost'] * edge_vars[f"y{u}_{v}"] for u, v in E2) -
                          quicksum(labor_costs[f"l{node}"] * U_labor[f"U_l{node}"] for node in range(self.num_labor_resources)))
        
        # Constraints
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1, name=f"C_{u}_{v}")
            else:
                model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1, name=f"C_{u}_{v}")
        
        # Production yield constraints adhering to zoning limits
        for node in G.nodes:
            model.addCons(production_yield[f"Y{node}"] <= zone_limits[node], name=f"Zone_Limit_{node}")

        # Set Partitioning constraints
        for node in G.nodes:
            model.addCons(quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2 if u == node or v == node) == node_vars[f"x{node}"], name=f"Partition_{node}")

        # Knapsack constraints on nodes
        for node in G.nodes:
            model.addCons(quicksum(edge_vars[f"y{u}_{v}"] for u, v in G.edges if u == node or v == node) <= node_capacities[node], name=f"Knapsack_Node_{node}")

        # Additional edge constraints
        model.addCons(quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2) <= max_removable_edges, name="Max_Removable_Edges")
        
        # Labor constraints
        for node in range(self.num_labor_resources):
            model.addCons(quicksum(production_yield[f"Y{n}"] for n in G.nodes) >= labor_quotas[f"l{node}"], name=f"Labor_Quota_{node}")
            model.addCons(U_labor[f"U_l{node}"] <= max_working_hours[f"l{node}"], name=f"Max_Working_Hours_{node}")
            model.addCons(U_labor[f"U_l{node}"] <= bal_factor[f"B_l{node}"], name=f"Labor_Balance_{node}")
        
        # Budget constraint for labor costs
        model.addCons(quicksum(labor_costs[f"l{node}"] * U_labor[f"U_l{node}"] for node in range(self.num_labor_resources)) <= self.labor_budget, name="Labor_Budget")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 135,
        'max_n': 219,
        'set_type': 'SET1',
        'set_param': 3000.0,
        'alpha': 0.59,
        'ba_m': 7,
        'min_capacity': 67,
        'max_capacity': 1177,
        'min_removable_edges': 330,
        'max_removable_edges': 1260,
        'min_zone_limit': 375,
        'max_zone_limit': 3000,
        'num_labor_resources': 100,
        'min_labor_cost': 100,
        'max_labor_cost': 2000,
        'min_quota': 20,
        'max_quota': 450,
        'min_working_hours': 105,
        'max_working_hours': 500,
        'labor_budget': 5000,
    }
    
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")