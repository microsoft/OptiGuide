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

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.incompatibility_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_zones(self, G):
        zones = list(nx.find_cliques(G.to_undirected()))
        return zones

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_network_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        zones = self.create_zones(G)

        network_capacity = {node: np.random.randint(150, 300) for node in G.nodes}
        operational_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}

        service_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            service_scenarios[s]['demand'] = {node: np.random.normal(G.nodes[node]['demand'], G.nodes[node]['demand'] * self.NetworkDemandVariation)
                                              for node in G.nodes}
            service_scenarios[s]['bandwidth'] = {(u, v): np.random.normal(G[u][v]['bandwidth'], G[u][v]['bandwidth'] * self.BandwidthVariation)
                                                 for u, v in G.edges}
            service_scenarios[s]['network_capacity'] = {node: np.random.normal(network_capacity[node], network_capacity[node] * self.NetworkCapacityVariation)
                                                       for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'zones': zones,
            'network_capacity': network_capacity,
            'operational_cost': operational_cost,
            'service_scenarios': service_scenarios
        }

    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        network_capacity = instance['network_capacity']
        operational_cost = instance['operational_cost']
        service_scenarios = instance['service_scenarios']

        model = Model("NetworkOptimization")
        NetworkNode_vars = {f"NetworkNode{node}": model.addVar(vtype="B", name=f"NetworkNode{node}") for node in G.nodes}
        NetworkRoute_vars = {f"NetworkRoute{u}_{v}": model.addVar(vtype="B", name=f"NetworkRoute{u}_{v}") for u, v in G.edges}

        # Scenario-specific variables
        demand_vars = {s: {f"NetworkNode{node}_s{s}": model.addVar(vtype="B", name=f"NetworkNode{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        bandwidth_vars = {s: {f"NetworkRoute{u}_{v}_s{s}": model.addVar(vtype="B", name=f"NetworkRoute{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.no_of_scenarios)}
        capacity_vars = {s: {f"NetworkCapacity{node}_s{s}": model.addVar(vtype="B", name=f"NetworkCapacity{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}

        # Objective function
        objective_expr = quicksum(
            service_scenarios[s]['demand'][node] * demand_vars[s][f"NetworkNode{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            service_scenarios[s]['bandwidth'][(u, v)] * bandwidth_vars[s][f"NetworkRoute{u}_{v}_s{s}"]
            for s in range(self.no_of_scenarios) for u, v in E_invalid
        )

        objective_expr -= quicksum(
            service_scenarios[s]['network_capacity'][node] * service_scenarios[s]['demand'][node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            operational_cost[(u, v)] * NetworkRoute_vars[f"NetworkRoute{u}_{v}"]
            for u, v in G.edges
        )

        # Constraint: Compatible Network Headquarters Placement
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

        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    demand_vars[s][f"NetworkNode{node}_s{s}"] == NetworkNode_vars[f"NetworkNode{node}"],
                    name=f"NetworkDemand_{node}_s{s}"
                )
                model.addCons(
                    capacity_vars[s][f"NetworkCapacity{node}_s{s}"] == NetworkNode_vars[f"NetworkNode{node}"],
                    name=f"NetworkCapacity_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    bandwidth_vars[s][f"NetworkRoute{u}_{v}_s{s}"] == NetworkRoute_vars[f"NetworkRoute{u}_{v}"],
                    name=f"BandwidthConstraint_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 66,
        'max_nodes': 153,
        'network_probability': 0.31,
        'incompatibility_rate': 0.1,
        'no_of_scenarios': 7,
        'NetworkDemandVariation': 0.17,
        'BandwidthVariation': 0.52,
        'NetworkCapacityVariation': 0.24,
    }

    network_opt = NetworkOptimization(parameters, seed=seed)
    instance = network_opt.get_instance()
    solve_status, solve_time = network_opt.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")