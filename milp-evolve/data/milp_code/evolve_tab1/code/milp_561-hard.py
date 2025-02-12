import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class FleetManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_logistics_network(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.edge_probability, directed=True, seed=self.seed)
        return G

    def generate_node_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.poisson(150, 1)[0]
        for u, v in G.edges:
            G[u][v]['travel_time'] = np.random.weibull(1.5)
            G[u][v]['capacity'] = np.random.randint(50, 300)

    def generate_truck_incompatibility(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.incompatibility_rate:
                E_invalid.add(edge)
        return E_invalid

    def get_instance(self):
        G = self.generate_logistics_network()
        self.generate_node_data(G)
        E_invalid = self.generate_truck_incompatibility(G)

        node_capacity = {node: np.random.randint(200, 800) for node in G.nodes}
        travel_cost = {(u, v): np.random.uniform(20.0, 100.0) for u, v in G.edges}

        dist_scenarios = [{} for _ in range(self.num_scenarios)]
        for s in range(self.num_scenarios):
            dist_scenarios[s]['demand'] = {node: np.random.poisson(G.nodes[node]['demand']) for node in G.nodes}
            dist_scenarios[s]['travel_time'] = {(u, v): np.random.weibull(1.5) for u, v in G.edges}
            dist_scenarios[s]['node_capacity'] = {node: np.random.normal(node_capacity[node], node_capacity[node] * self.capacity_variation) for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'node_capacity': node_capacity,
            'travel_cost': travel_cost,
            'dist_scenarios': dist_scenarios,
        }

    def solve(self, instance):
        G, E_invalid = instance['G'], instance['E_invalid']
        node_capacity = instance['node_capacity']
        travel_cost = instance['travel_cost']
        dist_scenarios = instance['dist_scenarios']

        model = Model("FleetManagement")
        
        # Define variables
        truck_vars = {node: model.addVar(vtype="B", name=f"Truck{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}

        # Objective function
        objective_expr = quicksum(
            dist_scenarios[s]['demand'][node] * truck_vars[node]
            for s in range(self.num_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            travel_cost[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges if (u, v) not in E_invalid
        )

        model.setObjective(objective_expr, "maximize")

        # Constraints
        for u, v in G.edges:
            model.addCons(
                truck_vars[u] + truck_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"RouteFlow_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                quicksum(route_vars[f"Route_{u}_{v}"] for u, v in G.edges if u == node or v == node) <= node_capacity[node],
                name=f"NodeCapacity_{node}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 126,
        'max_nodes': 300,
        'edge_probability': 0.52,
        'incompatibility_rate': 0.75,
        'num_scenarios': 1000,
        'capacity_variation': 0.59,
    }

    fleet_management = FleetManagement(parameters, seed=seed)
    instance = fleet_management.get_instance()
    solve_status, solve_time = fleet_management.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")