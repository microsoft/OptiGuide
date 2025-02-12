import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NDP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_network(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, directed=True, seed=self.seed)
        return G

    def generate_network_data(self, G):
        for node in G.nodes:
            G.nodes[node]['traffic'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['installation_cost'] = np.random.randint(1, 20)
            G[u][v]['network_strength'] = np.random.uniform(0.5, 1.5)

    def generate_hub_capacities(self, G):
        hub_capacity = {node: np.random.randint(100, 1000) for node in G.nodes}
        return hub_capacity

    def generate_instance(self):
        G = self.generate_random_network()
        self.generate_network_data(G)
        hub_capacity = self.generate_hub_capacities(G)

        network_strength = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}

        scenarios = [{} for _ in range(self.n_scenarios)]
        for s in range(self.n_scenarios):
            scenarios[s]['traffic'] = {node: np.random.normal(G.nodes[node]['traffic'], G.nodes[node]['traffic'] * self.traffic_deviation)
                                       for node in G.nodes}
            scenarios[s]['installation_cost'] = {(u, v): np.random.normal(G[u][v]['installation_cost'], G[u][v]['installation_cost'] * self.cost_deviation)
                                                 for u, v in G.edges}
            scenarios[s]['hub_capacity'] = {node: np.random.normal(hub_capacity[node], hub_capacity[node] * self.capacity_deviation)
                                            for node in G.nodes}

        return {
            'G': G,
            'hub_capacity': hub_capacity,
            'network_strength': network_strength,
            'scenarios': scenarios
        }
    
    def solve(self, instance):
        G = instance['G']
        hub_capacity = instance['hub_capacity']
        network_strength = instance['network_strength']
        scenarios = instance['scenarios']
        
        model = Model("NDP")
        hub_vars = {f"h{node}": model.addVar(vtype="B", name=f"h{node}") for node in G.nodes}
        data_flow_vars = {f"f{u}_{v}": model.addVar(vtype="C", name=f"f{u}_{v}") for u, v in G.edges}

        traffic_vars = {s: {f"t{node}_s{s}": model.addVar(vtype="C", name=f"t{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}
        installation_cost_vars = {s: {f"ic{u}_{v}_s{s}": model.addVar(vtype="C", name=f"ic{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.n_scenarios)}
        hub_capacity_vars = {s: {f"cap{node}_s{s}": model.addVar(vtype="C", name=f"cap{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}

        objective_expr = quicksum(
            scenarios[s]['traffic'][node] * traffic_vars[s][f"t{node}_s{s}"]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            scenarios[s]['installation_cost'][(u, v)] * installation_cost_vars[s][f"ic{u}_{v}_s{s}"]
            for s in range(self.n_scenarios) for u, v in G.edges
        )

        objective_expr -= quicksum(
            scenarios[s]['hub_capacity'][node] * scenarios[s]['traffic'][node]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            network_strength[(u, v)] * data_flow_vars[f"f{u}_{v}"]
            for u, v in G.edges
        )

        for node in G.nodes:
            model.addCons(
                quicksum(traffic_vars[s][f"t{node}_s{s}"] for s in range(self.n_scenarios)) <= hub_vars[f"h{node}"] * hub_capacity[node],
                name=f"HubCapacity_{node}"
            )
        
        for u, v in G.edges:
            model.addCons(
                hub_vars[f"h{u}"] + hub_vars[f"h{v}"] <= 1 + data_flow_vars[f"f{u}_{v}"],
                name=f"Connectivity_{u}_{v}"
            )
            model.addCons(
                hub_vars[f"h{u}"] + hub_vars[f"h{v}"] >= 2 * data_flow_vars[f"f{u}_{v}"],
                name=f"Flow_{u}_{v}"
            )

        max_delay = model.addVar(vtype="C", name="max_delay")
        model.addCons(
            max_delay <= self.max_delay_limit,
            name="Max_delay_limit"
        )
        
        for s in range(self.n_scenarios):
            for node in G.nodes:
                model.addCons(
                    traffic_vars[s][f"t{node}_s{s}"] == max_delay, 
                    name=f"RobustTraffic_{node}_s{s}"
                )
                model.addCons(
                    hub_capacity_vars[s][f"cap{node}_s{s}"] == hub_vars[f"h{node}"],
                    name=f"RobustHubCapacity_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    installation_cost_vars[s][f"ic{u}_{v}_s{s}"] == data_flow_vars[f"f{u}_{v}"],
                    name=f"RobustInstallationCost_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 110,
        'max_nodes': 461,
        'er_prob': 0.45,
        'max_delay_limit': 1000,
        'n_scenarios': 45,
        'traffic_deviation': 0.38,
        'cost_deviation': 0.17,
        'capacity_deviation': 0.17,
    }

    ndp = NDP(parameters, seed=seed)
    instance = ndp.generate_instance()
    solve_status, solve_time = ndp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")