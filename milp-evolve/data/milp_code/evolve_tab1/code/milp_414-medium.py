import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WDN:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_city_network(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.network_prob, directed=True, seed=self.seed)
        return G

    def generate_distribution_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(100, 1000)

        for u, v in G.edges:
            G[u][v]['delivery_time'] = np.random.randint(1, 3)
            G[u][v]['capacity'] = np.random.randint(5, 15)

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
        G = self.generate_city_network()
        self.generate_distribution_data(G)
        E_invalid = self.generate_exclusion_data(G)
        zones = self.create_zones(G)

        hub_capacity = {node: np.random.randint(100, 500) for node in G.nodes}
        delivery_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}

        distribution_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            distribution_scenarios[s]['demand'] = {node: np.random.normal(G.nodes[node]['demand'], G.nodes[node]['demand'] * self.demand_variation)
                                                   for node in G.nodes}
            distribution_scenarios[s]['delivery_time'] = {(u, v): np.random.normal(G[u][v]['delivery_time'], G[u][v]['delivery_time'] * self.time_variation)
                                                          for u, v in G.edges}
            distribution_scenarios[s]['hub_capacity'] = {node: np.random.normal(hub_capacity[node], hub_capacity[node] * self.capacity_variation)
                                                         for node in G.nodes}

            # New data for shortages and excesses
            distribution_scenarios[s]['shortages'] = {node: np.random.gamma(shape=2.0, scale=50) for node in G.nodes}
            distribution_scenarios[s]['excesses'] = {node: np.random.gamma(shape=2.0, scale=20) for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'zones': zones, 
            'hub_capacity': hub_capacity,
            'delivery_cost': delivery_cost,
            'distribution_scenarios': distribution_scenarios
        }

    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        hub_capacity = instance['hub_capacity']
        delivery_cost = instance['delivery_cost']
        distribution_scenarios = instance['distribution_scenarios']

        model = Model("WDN")
        hub_vars = {f"Hub{node}": model.addVar(vtype="B", name=f"Hub{node}") for node in G.nodes}
        truck_route_vars = {f"TruckRoute{u}_{v}": model.addVar(vtype="B", name=f"TruckRoute{u}_{v}") for u, v in G.edges}
        maintenance_budget = model.addVar(vtype="C", name="maintenance_budget")

        # Scenario-specific variables
        demand_vars = {s: {f"Hub{node}_s{s}": model.addVar(vtype="B", name=f"Hub{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        delivery_time_vars = {s: {f"TruckRoute{u}_{v}_s{s}": model.addVar(vtype="B", name=f"TruckRoute{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.no_of_scenarios)}
        capacity_vars = {s: {f"Capacity{node}_s{s}": model.addVar(vtype="B", name=f"Capacity{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        shortage_vars = {s: {f"Shortage{node}_s{s}": model.addVar(vtype="C", name=f"Shortage{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        excess_vars = {s: {f"Excess{node}_s{s}": model.addVar(vtype="C", name=f"Excess{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}

        M = 10000  # Big M value, should be larger than any possible demand/capacity values

        # Objective function
        objective_expr = quicksum(
            distribution_scenarios[s]['shortages'][node] * shortage_vars[s][f"Shortage{node}_s{s}"] +
            distribution_scenarios[s]['excesses'][node] * excess_vars[s][f"Excess{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            distribution_scenarios[s]['delivery_time'][(u, v)] * delivery_time_vars[s][f"TruckRoute{u}_{v}_s{s}"]
            for s in range(self.no_of_scenarios) for u, v in E_invalid
        )

        objective_expr -= quicksum(
            distribution_scenarios[s]['hub_capacity'][node] * distribution_scenarios[s]['demand'][node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            delivery_cost[(u, v)] * truck_route_vars[f"TruckRoute{u}_{v}"]
            for u, v in G.edges
        )

        # Zone constraints
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(hub_vars[f"Hub{node}"] for node in zone) <= 1,
                name=f"ZoneGroup_{i}"
            )

        # Big M constraints for delivery flows
        for u, v in G.edges:    
            model.addCons(
                hub_vars[f"Hub{u}"] + hub_vars[f"Hub{v}"] - truck_route_vars[f"TruckRoute{u}_{v}"] <= 1,
                name=f"DeliveryFlow_{u}_{v}"
            )
            model.addCons(
                2 * truck_route_vars[f"TruckRoute{u}_{v}"] - hub_vars[f"Hub{u}"] - hub_vars[f"Hub{v}"] <= 0,
                name=f"DeliveryFlow_{u}_{v}_other"
            )
            # Big M constraints
            model.addCons(
                hub_capacity[u] + M * (1 - truck_route_vars[f"TruckRoute{u}_{v}"]) >= G[u][v]['capacity'],
                name=f"BigM_Flow_Start_{u}_{v}"
            )
            model.addCons(
                hub_capacity[v] + M * (1 - truck_route_vars[f"TruckRoute{u}_{v}"]) >= G[u][v]['capacity'],
                name=f"BigM_Flow_End_{u}_{v}"
            )
        
        model.addCons(
            maintenance_budget <= self.maintenance_hours,
            name="Budget_Limit"
        )

        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    demand_vars[s][f"Hub{node}_s{s}"] == hub_vars[f"Hub{node}"],
                    name=f"DemandScenario_{node}_s{s}"
                )
                model.addCons(
                    capacity_vars[s][f"Capacity{node}_s{s}"] == hub_vars[f"Hub{node}"],
                    name=f"CapacityAvailable_{node}_s{s}"
                )
                # Shortage and Excess constraints
                model.addCons(
                    shortage_vars[s][f"Shortage{node}_s{s}"] >= distribution_scenarios[s]['demand'][node] - hub_vars[f"Hub{node}"],
                    name=f"ShortageConstraint_{node}_s{s}"
                )
                model.addCons(
                    excess_vars[s][f"Excess{node}_s{s}"] >= hub_vars[f"Hub{node}"] - distribution_scenarios[s]['demand'][node],
                    name=f"ExcessConstraint_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    delivery_time_vars[s][f"TruckRoute{u}_{v}_s{s}"] == truck_route_vars[f"TruckRoute{u}_{v}"],
                    name=f"RouteConstraint_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 132,
        'max_nodes': 153,
        'network_prob': 0.17,
        'exclusion_rate': 0.63,
        'maintenance_hours': 1680,
        'no_of_scenarios': 84,
        'demand_variation': 0.66,
        'time_variation': 0.6,
        'capacity_variation': 0.69,
    }

    wdn = WDN(parameters, seed=seed)
    instance = wdn.get_instance()
    solve_status, solve_time = wdn.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")