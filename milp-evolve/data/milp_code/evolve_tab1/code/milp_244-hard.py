import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WLP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_customers = np.random.randint(self.min_customers, self.max_customers)
        G = nx.erdos_renyi_graph(n=n_customers, p=self.er_prob, directed=True, seed=self.seed)
        return G

    def generate_customers_resources_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['transport_cost'] = np.random.randint(1, 20)
            G[u][v]['capacity'] = np.random.randint(1, 10)

    def generate_supplier_capacities(self, G):
        supplier_capacity = {node: np.random.randint(100, 1000) for node in G.nodes}
        return supplier_capacity

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_customers_resources_data(G)
        supplier_capacity = self.generate_supplier_capacities(G)

        # Generate transportation complication and supply-demand compatibility
        supply_demand_complexity = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}
        supply_compatibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}

        # Stochastic scenarios data generation
        scenarios = [{} for _ in range(self.n_scenarios)]
        for s in range(self.n_scenarios):
            scenarios[s]['demand'] = {node: np.random.normal(G.nodes[node]['demand'], G.nodes[node]['demand'] * self.demand_deviation)
                                      for node in G.nodes}
            scenarios[s]['transport_cost'] = {(u, v): np.random.normal(G[u][v]['transport_cost'], G[u][v]['transport_cost'] * self.cost_deviation)
                                              for u, v in G.edges}
            scenarios[s]['supplier_capacity'] = {node: np.random.normal(supplier_capacity[node], supplier_capacity[node] * self.capacity_deviation)
                                                 for node in G.nodes}

        return {
            'G': G,
            'supplier_capacity': supplier_capacity, 
            'supply_demand_complexity': supply_demand_complexity,
            'supply_compatibility': supply_compatibility,
            'scenarios': scenarios
        }
    
    def solve(self, instance):
        G = instance['G']
        supplier_capacity = instance['supplier_capacity']
        supply_demand_complexity = instance['supply_demand_complexity']
        supply_compatibility = instance['supply_compatibility']
        scenarios = instance['scenarios']
        
        model = Model("WLP")
        warehouse_vars = {f"w{node}": model.addVar(vtype="B", name=f"w{node}") for node in G.nodes}
        supply_vars = {f"s{u}_{v}": model.addVar(vtype="B", name=f"s{u}_{v}") for u, v in G.edges}

        # Scenario-specific variables
        demand_vars = {s: {f"c{node}_s{s}": model.addVar(vtype="C", name=f"c{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}
        transport_cost_vars = {s: {f"t{u}_{v}_s{s}": model.addVar(vtype="C", name=f"t{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.n_scenarios)}
        capacity_vars = {s: {f"cap{node}_s{s}": model.addVar(vtype="C", name=f"cap{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}

        objective_expr = quicksum(
            scenarios[s]['demand'][node] * demand_vars[s][f"c{node}_s{s}"]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            scenarios[s]['transport_cost'][(u, v)] * transport_cost_vars[s][f"t{u}_{v}_s{s}"]
            for s in range(self.n_scenarios) for u, v in G.edges
        )

        objective_expr -= quicksum(
            scenarios[s]['supplier_capacity'][node] * scenarios[s]['demand'][node]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            supply_demand_complexity[(u, v)] * supply_vars[f"s{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            supply_compatibility[(u, v)] * supply_vars[f"s{u}_{v}"]
            for u, v in G.edges
        )

        # Add constraints to ensure warehouse capacity is not exceeded
        for node in G.nodes:
            model.addCons(
                quicksum(demand_vars[s][f"c{node}_s{s}"] for s in range(self.n_scenarios)) <= warehouse_vars[f"w{node}"] * supplier_capacity[node],
                name=f"Capacity_{node}"
            )
        
        # Logical conditions to ensure unique customer assignments to warehouses
        for u, v in G.edges:
            model.addCons(
                warehouse_vars[f"w{u}"] + warehouse_vars[f"w{v}"] <= 1 + supply_vars[f"s{u}_{v}"],
                name=f"Sup1_{u}_{v}"
            )
            model.addCons(
                warehouse_vars[f"w{u}"] + warehouse_vars[f"w{v}"] >= 2 * supply_vars[f"s{u}_{v}"],
                name=f"Sup2_{u}_{v}"
            )

        weekly_budget = model.addVar(vtype="C", name="weekly_budget")
        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )
        
        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.n_scenarios):
            for node in G.nodes:
                model.addCons(
                    demand_vars[s][f"c{node}_s{s}"] == weekly_budget, # Ensure demand acts within budget
                    name=f"RobustDemand_{node}_s{s}"
                )
                model.addCons(
                    capacity_vars[s][f"cap{node}_s{s}"] == warehouse_vars[f"w{node}"],
                    name=f"RobustCapacity_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    transport_cost_vars[s][f"t{u}_{v}_s{s}"] == supply_vars[f"s{u}_{v}"],
                    name=f"RobustTransportCost_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_customers': 110,
        'max_customers': 615,
        'er_prob': 0.3,
        'weekly_budget_limit': 448,
        'n_scenarios': 60,
        'demand_deviation': 0.77,
        'cost_deviation': 0.48,
        'capacity_deviation': 0.45,
    }

    wlp = WLP(parameters, seed=seed)
    instance = wlp.generate_instance()
    solve_status, solve_time = wlp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")