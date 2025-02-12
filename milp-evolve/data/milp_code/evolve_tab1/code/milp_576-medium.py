import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WarehouseOptimization:
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

    def generate_logistics_data(self, G):
        for node in G.nodes:
            G.nodes[node]['capacity'] = np.random.randint(150, 300)
            G.nodes[node]['holding_cost'] = np.random.uniform(1.0, 5.0)

        for u, v in G.edges:
            G[u][v]['flow_cost'] = np.random.uniform(1.0, 5.0)

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.incompatibility_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_zones(self, G):
        zones = list(nx.find_cliques(G.to_undirected()))
        return zones

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_logistics_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        zones = self.create_zones(G)
        E2 = self.generate_removable_edges(G)

        warehouse_capacity = {node: np.random.randint(150, 300) for node in G.nodes}
        operational_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}

        demand_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            demand_scenarios[s]['demand'] = {node: np.random.normal(G.nodes[node]['capacity'], G.nodes[node]['capacity'] * self.DemandVariation)
                                             for node in G.nodes}
            demand_scenarios[s]['flow_cost'] = {(u, v): np.random.normal(G[u][v]['flow_cost'], G[u][v]['flow_cost'] * self.FlowCostVariation)
                                                for u, v in G.edges}
            demand_scenarios[s]['warehouse_capacity'] = {node: np.random.normal(warehouse_capacity[node], warehouse_capacity[node] * self.CapacityVariation)
                                                         for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'zones': zones,
            'warehouse_capacity': warehouse_capacity,
            'operational_cost': operational_cost,
            'demand_scenarios': demand_scenarios,
            'E2': E2
        }

    def solve(self, instance):
        G, E_invalid, zones, E2 = instance['G'], instance['E_invalid'], instance['zones'], instance['E2']
        warehouse_capacity = instance['warehouse_capacity']
        operational_cost = instance['operational_cost']
        demand_scenarios = instance['demand_scenarios']

        model = Model("WarehouseOptimization")
        WarehouseLocation_vars = {f"WarehouseLocation{node}": model.addVar(vtype="B", name=f"WarehouseLocation{node}") for node in G.nodes}
        ItemFlow_vars = {f"ItemFlow{u}_{v}": model.addVar(vtype="B", name=f"ItemFlow{u}_{v}") for u, v in G.edges}
        Removal_vars = {f"Removal{u}_{v}": model.addVar(vtype="B", name=f"Removal{u}_{v}") for u, v in E2}

        # Scenario-specific variables
        demand_vars = {s: {f"WarehouseLocation{node}_s{s}": model.addVar(vtype="B", name=f"WarehouseLocation{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        flow_vars = {s: {f"ItemFlow{u}_{v}_s{s}": model.addVar(vtype="B", name=f"ItemFlow{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.no_of_scenarios)}
        capacity_vars = {s: {f"WarehouseCapacity{node}_s{s}": model.addVar(vtype="B", name=f"WarehouseCapacity{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}

        # Objective function
        objective_expr = quicksum(
            demand_scenarios[s]['demand'][node] * demand_vars[s][f"WarehouseLocation{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            demand_scenarios[s]['flow_cost'][(u, v)] * flow_vars[s][f"ItemFlow{u}_{v}_s{s}"]
            for s in range(self.no_of_scenarios) for u, v in E_invalid
        )

        objective_expr -= quicksum(
            demand_scenarios[s]['warehouse_capacity'][node] * demand_scenarios[s]['demand'][node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            operational_cost[(u, v)] * ItemFlow_vars[f"ItemFlow{u}_{v}"]
            for u, v in G.edges
        )

        # Constraint: Warehouse Capacity Constraints
        for node in G.nodes:
            model.addCons(
                quicksum(demand_vars[s][f"WarehouseLocation{node}_s{s}"] for s in range(self.no_of_scenarios)) <= warehouse_capacity[node],
                name=f"Capacity_{node}"
            )

        # Constraint: Holding Cost Constraints
        for node in G.nodes:
            model.addCons(
                quicksum(demand_vars[s][f"WarehouseLocation{node}_s{s}"] * G.nodes[node]['holding_cost'] for s in range(self.no_of_scenarios)) <= warehouse_capacity[node],
                name=f"HoldingCost_{node}"
            )

        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    demand_vars[s][f"WarehouseLocation{node}_s{s}"] == WarehouseLocation_vars[f"WarehouseLocation{node}"],
                    name=f"Demand_{node}_s{s}"
                )
                model.addCons(
                    capacity_vars[s][f"WarehouseCapacity{node}_s{s}"] == WarehouseLocation_vars[f"WarehouseLocation{node}"],
                    name=f"Capacity_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    flow_vars[s][f"ItemFlow{u}_{v}_s{s}"] == ItemFlow_vars[f"ItemFlow{u}_{v}"],
                    name=f"Flow_{u}_{v}_s{s}"
                )

        # New Constraints: Clique Constraints
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(WarehouseLocation_vars[f"WarehouseLocation{node}"] for node in zone) <= 1,
                name=f"Clique_{i}"
            )

        # New Constraints: Edge Removal Constraints
        for u, v in E2:
            model.addCons(
                WarehouseLocation_vars[f"WarehouseLocation{u}"] + WarehouseLocation_vars[f"WarehouseLocation{v}"] - Removal_vars[f"Removal{u}_{v}"] <= 1,
                name=f"EdgeRemoval_{u}_{v}"
            )

        # Set objective
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 26,
        'max_nodes': 602,
        'network_probability': 0.17,
        'incompatibility_rate': 0.8,
        'no_of_scenarios': 24,
        'DemandVariation': 0.52,
        'FlowCostVariation': 0.45,
        'CapacityVariation': 0.38,
        'alpha': 0.17,
    }

    warehouse_opt = WarehouseOptimization(parameters, seed=seed)
    instance = warehouse_opt.get_instance()
    solve_status, solve_time = warehouse_opt.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")