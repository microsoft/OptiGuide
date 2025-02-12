import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexGISPWithRetail:
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
        if self.graph_type == 'ER':
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        elif self.graph_type == 'BA':
            G = nx.barabasi_albert_graph(n=n_nodes, m=self.barabasi_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
            G.nodes[node]['penalty'] = np.random.randint(1, 50)

        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.cost_param)

    def generate_retail_data(self, num_hubs, num_stores):
        transport_cost = np.random.randint(50, 300, size=(num_stores, num_hubs))
        opening_costs_hub = np.random.randint(1000, 3000, size=num_hubs)
        store_demands = np.random.randint(100, 500, size=num_stores)
        hub_capacity = np.random.randint(5000, 10000, size=num_hubs)
        max_acceptable_distance = np.random.randint(50, 200)
        store_distances_from_hubs = np.random.randint(10, 500, size=(num_stores, num_hubs))
        
        return {
            'num_hubs': num_hubs,
            'num_stores': num_stores,
            'transport_cost': transport_cost,
            'opening_costs_hub': opening_costs_hub,
            'store_demands': store_demands,
            'hub_capacity': hub_capacity,
            'store_distances_from_hubs': store_distances_from_hubs,
            'max_acceptable_distance': max_acceptable_distance
        }

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        res = {'G': G}

        for u, v in G.edges:
            break_points = np.linspace(0, G[u][v]['cost'], self.num_pieces+1)
            slopes = np.diff(break_points)
            res[f'break_points_{u}_{v}'] = break_points
            res[f'slopes_{u}_{v}'] = slopes
        
        num_hubs = np.random.randint(self.min_hubs, self.max_hubs)
        num_stores = len(G.nodes)
        retail_data = self.generate_retail_data(num_hubs, num_stores)

        res.update(retail_data)
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        num_hubs = instance['num_hubs']
        num_stores = instance['num_stores']
        transport_cost = instance['transport_cost']
        opening_costs_hub = instance['opening_costs_hub']
        store_demands = instance['store_demands']
        hub_capacity = instance['hub_capacity']
        store_distances_from_hubs = instance['store_distances_from_hubs']
        max_acceptable_distance = instance['max_acceptable_distance']
        
        model = Model("ComplexGISPWithRetail")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piece_vars = {f"t{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"t{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_pieces)}

        # Retail Network related variables
        Number_of_Hubs = {j: model.addVar(vtype="B", name=f"Number_of_Hubs_{j}") for j in range(num_hubs)}
        Customer_Satisfaction = {(i, j): model.addVar(vtype="B", name=f"Customer_Satisfaction_{i}_{j}") for i in range(num_stores) for j in range(num_hubs)}

        # Modified objective function
        node_revenue = quicksum(G.nodes[node]['revenue'] * node_vars[f"x{node}"] for node in G.nodes)
        edge_cost = quicksum(instance[f'slopes_{u}_{v}'][k] * piece_vars[f"t{u}_{v}_{k}"] for u, v in G.edges for k in range(self.num_pieces))
        transport_cost_total = quicksum(Customer_Satisfaction[i, j] * transport_cost[i, j] for i in range(num_stores) for j in range(num_hubs))
        opening_costs_total = quicksum(Number_of_Hubs[j] * opening_costs_hub[j] for j in range(num_hubs))

        objective_expr = node_revenue - edge_cost - transport_cost_total - opening_costs_total

        # Applying Piecewise Linear Function Constraints
        M = 10000  # Big M constant, should be large enough
        for u, v in G.edges:
            model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] * M <= 1, name=f"C_{u}_{v}")
                
        for u, v in G.edges:
            model.addCons(quicksum(piece_vars[f"t{u}_{v}_{k}"] for k in range(self.num_pieces)) == edge_vars[f'y{u}_{v}'])

            for k in range(self.num_pieces):
                model.addCons(piece_vars[f"t{u}_{v}_{k}"] <= instance[f'break_points_{u}_{v}'][k+1] - instance[f'break_points_{u}_{v}'][k] + (1 - edge_vars[f'y{u}_{v}']) * M)

        # Retail Network Constraints
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

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 81,
        'max_n': 1024,
        'er_prob': 0.52,
        'graph_type': 'BA',
        'barabasi_m': 18,
        'cost_param': 3000.0,
        'num_pieces': 7,
        'min_hubs': 10,
        'max_hubs': 25,
    }

    gisp = ComplexGISPWithRetail(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")