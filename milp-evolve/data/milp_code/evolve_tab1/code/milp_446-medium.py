import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EMS_Optimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.zone_prob, directed=True, seed=self.seed)
        return G

    def generate_ems_data(self, G):
        for node in G.nodes:
            G.nodes[node]['emergency_calls'] = np.random.randint(5, 100)

        for u, v in G.edges:
            G[u][v]['response_time'] = np.random.randint(1, 5)
            G[u][v]['capacity'] = np.random.randint(1, 10)

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_ems_data(G)
        zones = list(nx.find_cliques(G.to_undirected()))

        ems_capacity = {node: np.random.randint(5, 50) for node in G.nodes}
        operational_cost = {(u, v): np.random.uniform(2.0, 10.0) for u, v in G.edges}

        service_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            service_scenarios[s]['emergency_calls'] = {node: np.random.normal(G.nodes[node]['emergency_calls'], G.nodes[node]['emergency_calls'] * self.demand_variation)
                                                       for node in G.nodes}
            service_scenarios[s]['ems_capacity'] = {node: np.random.normal(ems_capacity[node], ems_capacity[node] * self.capacity_variation)
                                                    for node in G.nodes}

        return {
            'G': G,
            'zones': zones,
            'ems_capacity': ems_capacity,
            'operational_cost': operational_cost,
            'service_scenarios': service_scenarios
        }

    def solve(self, instance):
        G, zones = instance['G'], instance['zones']
        ems_capacity = instance['ems_capacity']
        operational_cost = instance['operational_cost']
        service_scenarios = instance['service_scenarios']

        model = Model("EMS_Optimization")
        dispatch_units_vars = {f"DispatchUnit{node}": model.addVar(vtype="B", name=f"DispatchUnit{node}") for node in G.nodes}
        emergency_route_vars = {f"EmergencyRoute{u}_{v}": model.addVar(vtype="B", name=f"EmergencyRoute{u}_{v}") for u, v in G.edges}

        # Scenario-specific variables
        emergency_call_vars = {s: {f"DispatchUnit{node}_s{s}": model.addVar(vtype="B", name=f"DispatchUnit{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}

        # Objective function
        objective_expr = quicksum(
            service_scenarios[s]['emergency_calls'][node] * emergency_call_vars[s][f"DispatchUnit{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            operational_cost[(u, v)] * emergency_route_vars[f"EmergencyRoute{u}_{v}"]
            for u, v in G.edges
        )

        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(dispatch_units_vars[f"DispatchUnit{node}"] for node in zone) <= 1,
                name=f"ZoneIncompatibility_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                dispatch_units_vars[f"DispatchUnit{u}"] + dispatch_units_vars[f"DispatchUnit{v}"] <= 1 + emergency_route_vars[f"EmergencyRoute{u}_{v}"],
                name=f"EmergencyFlow_{u}_{v}"
            )
            model.addCons(
                dispatch_units_vars[f"DispatchUnit{u}"] + dispatch_units_vars[f"DispatchUnit{v}"] >= 2 * emergency_route_vars[f"EmergencyRoute{u}_{v}"],
                name=f"EmergencyFlow_{u}_{v}_other"
            )

        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    emergency_call_vars[s][f"DispatchUnit{node}_s{s}"] == dispatch_units_vars[f"DispatchUnit{node}"],
                    name=f"NodalDemand_{node}_s{s}"
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
        'zone_prob': 0.24,
        'no_of_scenarios': 5,  # Further reduced number of scenarios
        'demand_variation': 0.45,
        'capacity_variation': 0.1,
    }

    ems_opt = EMS_Optimization(parameters, seed=seed)
    instance = ems_opt.get_instance()
    solve_status, solve_time = ems_opt.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")