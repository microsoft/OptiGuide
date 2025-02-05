import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.network_probability, directed=True, seed=self.seed)
        return G

    def generate_network_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(100, 500)

        for u, v in G.edges:
            G[u][v]['bandwidth'] = np.random.randint(10, 50)
            G[u][v]['cost'] = np.random.uniform(1.0, 5.0)

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_network_data(G)
        zones = list(nx.find_cliques(G.to_undirected()))
        network_capacity = {node: np.random.randint(150, 300) for node in G.nodes}
        operational_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}

        return {
            'G': G,
            'zones': zones,
            'network_capacity': network_capacity,
            'operational_cost': operational_cost
        }

    def solve(self, instance):
        G, zones = instance['G'], instance['zones']
        network_capacity = instance['network_capacity']
        operational_cost = instance['operational_cost']

        model = Model("NetworkOptimization")
        NetworkNode_vars = {f"NetworkNode{node}": model.addVar(vtype="B", name=f"NetworkNode{node}") for node in G.nodes}
        NetworkRoute_vars = {f"NetworkRoute{u}_{v}": model.addVar(vtype="B", name=f"NetworkRoute{u}_{v}") for u, v in G.edges}

        # Objective function
        objective_expr = quicksum(
            G.nodes[node]['demand'] * NetworkNode_vars[f"NetworkNode{node}"] for node in G.nodes
        ) - quicksum(
            operational_cost[(u, v)] * NetworkRoute_vars[f"NetworkRoute{u}_{v}"] for u, v in G.edges
        )

        # Constraints
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(NetworkNode_vars[f"NetworkNode{node}"] for node in zone) <= 1,
                name=f"HeadquarterSetup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                NetworkNode_vars[f"NetworkNode{u}"] + NetworkNode_vars[f"NetworkNode{v}"] <= 1 + NetworkRoute_vars[f"NetworkRoute{u}_{v}"],
                name=f"NetworkFlow_{u}_{v}"
            )
            model.addCons(
                NetworkNode_vars[f"NetworkNode{u}"] + NetworkNode_vars[f"NetworkNode{v}"] >= 2 * NetworkRoute_vars[f"NetworkRoute{u}_{v}"],
                name=f"NetworkFlow_{u}_{v}_other"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 39,
        'max_nodes': 602,
        'network_probability': 0.17,
    }
    network_opt = NetworkOptimization(parameters, seed=seed)
    instance = network_opt.get_instance()
    solve_status, solve_time = network_opt.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")