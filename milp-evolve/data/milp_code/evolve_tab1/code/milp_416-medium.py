import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ShippingPortOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_port_network(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.network_prob, directed=True, seed=self.seed)
        return G

    def generate_distribution_data(self, G):
        for node in G.nodes:
            G.nodes[node]['cargo_weight'] = np.random.randint(10, 100)

        for u, v in G.edges:
            G[u][v]['handling_time'] = np.random.randint(1, 10)
            G[u][v]['capacity'] = np.random.randint(20, 50)

    def generate_exclusion_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_zones(self, G):
        zones = list(nx.find_cliques(G.to_undirected()))
        return zones

    def get_instance(self):
        G = self.generate_port_network()
        self.generate_distribution_data(G)
        E_invalid = self.generate_exclusion_data(G)
        zones = self.create_zones(G)

        trailer_count = {node: np.random.randint(3, 10) for node in G.nodes}
        handling_cost = {(u, v): np.random.uniform(10.0, 50.0) for u, v in G.edges}

        cargo_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            cargo_scenarios[s]['cargo_weight'] = {
                node: np.random.normal(G.nodes[node]['cargo_weight'], G.nodes[node]['cargo_weight'] * self.cargo_variation)
                for node in G.nodes
            }
            cargo_scenarios[s]['handling_time'] = {
                (u, v): np.random.normal(G[u][v]['handling_time'], G[u][v]['handling_time'] * self.time_variation)
                for u, v in G.edges
            }
            cargo_scenarios[s]['trailer_count'] = {
                node: np.random.normal(trailer_count[node], trailer_count[node] * self.trailer_variation)
                for node in G.nodes
            }

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'zones': zones, 
            'trailer_count': trailer_count, 
            'handling_cost': handling_cost,
            'cargo_scenarios': cargo_scenarios
        }
    
    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        trailer_count = instance['trailer_count']
        handling_cost = instance['handling_cost']
        cargo_scenarios = instance['cargo_scenarios']
        
        model = Model("ShippingPortOptimization")
        trailer_vars = {f"Trailer{node}": model.addVar(vtype="B", name=f"Trailer{node}") for node in G.nodes}
        merge_rate_vars = {f"MergeRate{u}_{v}": model.addVar(vtype="B", name=f"MergeRate{u}_{v}") for u, v in G.edges}
        zero_idle_time = model.addVar(vtype="C", name="ZeroIdleTime")

        # Scenario-specific variables
        cargo_weight_vars = {s: {f"Cargo{node}_s{s}": model.addVar(vtype="B", name=f"Cargo{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        handling_time_vars = {s: {f"Handler{u}_{v}_s{s}": model.addVar(vtype="B", name=f"Handler{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.no_of_scenarios)}
        capacity_vars = {s: {f"Capacity{node}_s{s}": model.addVar(vtype="B", name=f"Capacity{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}

        objective_expr = quicksum(
            cargo_scenarios[s]['cargo_weight'][node] * cargo_weight_vars[s][f"Cargo{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            cargo_scenarios[s]['handling_time'][(u, v)] * handling_time_vars[s][f"Handler{u}_{v}_s{s}"]
            for s in range(self.no_of_scenarios) for u, v in E_invalid
        )

        objective_expr -= quicksum(
            cargo_scenarios[s]['trailer_count'][node] * cargo_scenarios[s]['cargo_weight'][node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            handling_cost[(u, v)] * merge_rate_vars[f"MergeRate{u}_{v}"]
            for u, v in G.edges
        )

        # New constraints with Logical Conditions:
        
        # Ensure at most one trailer per connected group (clique)
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(trailer_vars[f"Trailer{node}"] for node in zone) <= 1,
                name=f"ZoneDeployment_{i}"
            )
        
        for u, v in G.edges:
            model.addCons(
                trailer_vars[f"Trailer{u}"] + trailer_vars[f"Trailer{v}"] <= 1 + merge_rate_vars[f"MergeRate{u}_{v}"],
                name=f"CargoFlow_{u}_{v}"
            )
            model.addCons(
                trailer_vars[f"Trailer{u}"] + trailer_vars[f"Trailer{v}"] >= 2 * merge_rate_vars[f"MergeRate{u}_{v}"],
                name=f"CargoFlow_{u}_{v}_additional"
            )

        model.addCons(
            zero_idle_time <= self.zero_idle_threshold,
            name="IdleTimeConstraint"
        )

        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    cargo_weight_vars[s][f"Cargo{node}_s{s}"] == trailer_vars[f"Trailer{node}"],
                    name=f"CargoScenario_{node}_s{s}"
                )
                model.addCons(
                    capacity_vars[s][f"Capacity{node}_s{s}"] == trailer_vars[f"Trailer{node}"],
                    name=f"CapacityUsage_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    handling_time_vars[s][f"Handler{u}_{v}_s{s}"] == merge_rate_vars[f"MergeRate{u}_{v}"],
                    name=f"HandlingConstraint_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 99,
        'max_nodes': 153,
        'network_prob': 0.17,
        'exclusion_rate': 0.59,
        'zero_idle_threshold': 1500,
        'no_of_scenarios': 42,
        'cargo_variation': 0.45,
        'time_variation': 0.24,
        'trailer_variation': 0.8,
    }

    port_optimization = ShippingPortOptimization(parameters, seed=seed)
    instance = port_optimization.get_instance()
    solve_status, solve_time = port_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")