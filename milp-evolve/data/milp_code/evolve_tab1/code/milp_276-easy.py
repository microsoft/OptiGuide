import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SCDN:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, directed=True, seed=self.seed)
        return G

    def generate_node_and_edge_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['transport_cost'] = np.random.randint(1, 20)
            G[u][v]['capacity'] = np.random.randint(1, 10)

    def generate_incompatible_edges(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E_invalid.add(edge)
        return E_invalid

    def find_hub_clusters(self, G):
        cliques = list(nx.find_cliques(G.to_undirected()))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_node_and_edge_data(G)
        E_invalid = self.generate_incompatible_edges(G)
        hubs = self.find_hub_clusters(G)

        manufacturing_cap = {node: np.random.randint(1, 40) for node in G.nodes}
        processing_cost = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}
        shipment_validity = {(u, v): np.random.randint(0, 2) for u, v in G.edges}

        orders = [(hub, np.random.uniform(50, 200)) for hub in hubs]

        scenarios = [{} for _ in range(self.n_scenarios)]
        for s in range(self.n_scenarios):
            scenarios[s]['demand'] = {node: np.random.normal(G.nodes[node]['demand'], G.nodes[node]['demand'] * self.demand_deviation)
                                      for node in G.nodes}
            scenarios[s]['transport_cost'] = {(u, v): np.random.normal(G[u][v]['transport_cost'], G[u][v]['transport_cost'] * self.cost_deviation)
                                               for u, v in G.edges}
            scenarios[s]['manufacturing_cap'] = {node: np.random.normal(manufacturing_cap[node], manufacturing_cap[node] * self.manufacturing_cap_deviation)
                                                 for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'hubs': hubs, 
            'manufacturing_cap': manufacturing_cap, 
            'processing_cost': processing_cost,
            'shipment_validity': shipment_validity,
            'orders': orders,
            'scenarios': scenarios
        }
    
    def solve(self, instance):
        G, E_invalid, hubs = instance['G'], instance['E_invalid'], instance['hubs']
        manufacturing_cap = instance['manufacturing_cap']
        processing_cost = instance['processing_cost']
        shipment_validity = instance['shipment_validity']
        orders = instance['orders']
        scenarios = instance['scenarios']
        
        model = Model("SCDN")
        hub_vars = {f"h{node}": model.addVar(vtype="B", name=f"h{node}") for node in G.nodes}
        transport_vars = {f"t{u}_{v}": model.addVar(vtype="B", name=f"t{u}_{v}") for u, v in G.edges}
        daily_budget = model.addVar(vtype="C", name="daily_budget")

        # New order variables
        order_vars = {i: model.addVar(vtype="B", name=f"Order_{i}") for i in range(len(orders))}

        # Scenario-specific variables
        demand_vars = {s: {f"h{node}_s{s}": model.addVar(vtype="B", name=f"h{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}
        transport_cost_vars = {s: {f"t{u}_{v}_s{s}": model.addVar(vtype="B", name=f"t{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.n_scenarios)}
        manuf_cap_vars = {s: {f"m{node}_s{s}": model.addVar(vtype="B", name=f"m{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}

        objective_expr = quicksum(
            scenarios[s]['demand'][node] * demand_vars[s][f"h{node}_s{s}"]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            scenarios[s]['transport_cost'][(u, v)] * transport_cost_vars[s][f"t{u}_{v}_s{s}"]
            for s in range(self.n_scenarios) for u, v in E_invalid
        )

        objective_expr -= quicksum(
            scenarios[s]['manufacturing_cap'][node] * scenarios[s]['demand'][node]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            processing_cost[(u, v)] * transport_vars[f"t{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            shipment_validity[(u, v)] * transport_vars[f"t{u}_{v}"]
            for u, v in G.edges
        )

        # New objective component to maximize order fulfilling
        objective_expr += quicksum(price * order_vars[i] for i, (bundle, price) in enumerate(orders))

        for i, hub in enumerate(hubs):
            model.addCons(
                quicksum(hub_vars[f"h{node}"] for node in hub) <= 1,
                name=f"HubGroup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                hub_vars[f"h{u}"] + hub_vars[f"h{v}"] <= 1 + transport_vars[f"t{u}_{v}"],
                name=f"NetworkFlowConstraint_{u}_{v}"
            )
            model.addCons(
                hub_vars[f"h{u}"] + hub_vars[f"h{v}"] >= 2 * transport_vars[f"t{u}_{v}"],
                name=f"NetworkFlowConstraint_{u}_{v}_other"
            )

        model.addCons(
            daily_budget <= self.daily_budget_limit,
            name="ContextHandling_budget_limit"
        )
    
        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.n_scenarios):
            for node in G.nodes:
                model.addCons(
                    demand_vars[s][f"h{node}_s{s}"] == hub_vars[f"h{node}"],
                    name=f"CustomerRequests_CareDemand_{node}_s{s}"
                )
                model.addCons(
                    manuf_cap_vars[s][f"m{node}_s{s}"] == hub_vars[f"h{node}"],
                    name=f"ManufacturingCap_Availability_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    transport_cost_vars[s][f"t{u}_{v}_s{s}"] == transport_vars[f"t{u}_{v}"],
                    name=f"NetworkFlowConstraint_Cost_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 132,
        'max_nodes': 153,
        'er_prob': 0.1,
        'alpha': 0.38,
        'daily_budget_limit': 280,
        'n_scenarios': 12,
        'demand_deviation': 0.45,
        'cost_deviation': 0.31,
        'manufacturing_cap_deviation': 0.52,
    }

    scdn = SCDN(parameters, seed=seed)
    instance = scdn.generate_instance()
    solve_status, solve_time = scdn.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")