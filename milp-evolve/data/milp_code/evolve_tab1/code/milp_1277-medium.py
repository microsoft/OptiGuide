import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class AerospaceSupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_components > 0 and self.n_suppliers > 0
        assert self.min_prod_cost >= 0 and self.max_prod_cost >= self.min_prod_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        
        # Production and transport costs
        production_costs = np.random.randint(self.min_prod_cost, self.max_prod_cost + 1, self.n_facilities)
        
        transport_costs_land = np.random.uniform(self.min_transport_cost, self.max_transport_cost, (self.n_facilities, self.n_components))
        transport_costs_air = np.random.uniform(self.min_transport_cost_air, self.max_transport_cost_air, (self.n_facilities, self.n_components))
        
        inventory_costs = np.random.uniform(self.min_inventory_cost, self.max_inventory_cost, self.n_components)
        lead_times = np.random.randint(self.min_lead_time, self.max_lead_time + 1, self.n_components)

        # Demand and capacity
        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_components).astype(int)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)

        suppliers = np.random.randint(self.min_suppliers, self.max_suppliers + 1, self.n_components)
        
        # Graph and travel parameters
        G = nx.erdos_renyi_graph(self.n_facilities, self.graph_density, seed=self.seed)
        travel_times = {(u, v): np.random.randint(10, 60) for u, v in G.edges}
        travel_costs_land = {(u, v): np.random.uniform(10.0, 60.0) for u, v in G.edges}
        travel_costs_air = {(u, v): np.random.uniform(30.0, 100.0) for u, v in G.edges}

        return {
            "production_costs": production_costs,
            "transport_costs_land": transport_costs_land,
            "transport_costs_air": transport_costs_air,
            "inventory_costs": inventory_costs,
            "lead_times": lead_times,
            "demands": demands,
            "capacities": capacities,
            "suppliers": suppliers,
            "graph": G,
            "travel_times": travel_times,
            "travel_costs_land": travel_costs_land,
            "travel_costs_air": travel_costs_air
        }

    def solve(self, instance):
        production_costs = instance["production_costs"]
        transport_costs_land = instance["transport_costs_land"]
        transport_costs_air = instance["transport_costs_air"]
        inventory_costs = instance["inventory_costs"]
        lead_times = instance["lead_times"]
        demands = instance["demands"]
        capacities = instance["capacities"]
        suppliers = instance["suppliers"]
        G = instance["graph"]
        travel_times = instance["travel_times"]
        travel_costs_land = instance["travel_costs_land"]
        travel_costs_air = instance["travel_costs_air"]

        model = Model("AerospaceSupplyChainOptimization")
        n_facilities = len(production_costs)
        n_components = len(transport_costs_land[0])
        n_suppliers = len(set(suppliers))

        BigM = self.bigM

        # Decision variables
        production_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars_land = {(f, c): model.addVar(vtype="B", name=f"Facility_{f}_Component_{c}_Land") for f in range(n_facilities) for c in range(n_components)}
        transport_vars_air = {(f, c): model.addVar(vtype="B", name=f"Facility_{f}_Component_{c}_Air") for f in range(n_facilities) for c in range(n_components)}
        route_vars = {(u, v): model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        assembly_vars = {c: model.addVar(vtype="B", name=f"Component_{c}_Assembled") for c in range(n_components)}

        inventory_vars = {c: model.addVar(vtype="I", lb=0, name=f"Inventory_{c}") for c in range(n_components)}
        delay_vars = {c: model.addVar(vtype="C", lb=0, name=f"Delay_{c}") for c in range(n_components)}
        backorder_vars = {c: model.addVar(vtype="C", lb=0, name=f"BackOrder_{c}") for c in range(n_components)}

        # Objective function
        model.setObjective(
            quicksum(production_costs[f] * production_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs_land[f, c] * transport_vars_land[f, c] for f in range(n_facilities) for c in range(n_components)) +
            quicksum(transport_costs_air[f, c] * transport_vars_air[f, c] for f in range(n_facilities) for c in range(n_components)) +
            quicksum(inventory_costs[c] * inventory_vars[c] for c in range(n_components)) +
            quicksum(travel_costs_land[(u, v)] * route_vars[(u, v)] for u, v in G.edges) +
            quicksum(travel_costs_air[(u, v)] * route_vars[(u, v)] for u, v in G.edges) +
            quicksum(self.delay_penalty * delay_vars[c] for c in range(n_components)) +
            quicksum(self.backorder_penalty * backorder_vars[c] for c in range(n_components))
            , "minimize"
        )

        # Constraints: Each component's demand should be met by at least one facility
        for c in range(n_components):
            model.addCons(quicksum(transport_vars_land[f, c] + transport_vars_air[f, c] for f in range(n_facilities)) >= 1, f"Component_{c}_Demand")

        # Constraints: Only open facilities can serve components
        for f in range(n_facilities):
            for c in range(n_components):
                model.addCons(transport_vars_land[f, c] <= production_vars[f], f"Facility_{f}_Serve_{c}_Land")
                model.addCons(transport_vars_air[f, c] <= production_vars[f], f"Facility_{f}_Serve_{c}_Air")

        # Constraints: Facilities cannot exceed their production capacity
        for f in range(n_facilities):
            model.addCons(quicksum(demands[c] * (transport_vars_land[f, c] + transport_vars_air[f, c]) for c in range(n_components)) <= capacities[f], f"Facility_{f}_Capacity")

        # Constraints: Only open facilities can route edges
        for u, v in G.edges:
            model.addCons(route_vars[(u, v)] <= production_vars[u], f"Route_{u}_{v}_facility_{u}")
            model.addCons(route_vars[(u, v)] <= production_vars[v], f"Route_{u}_{v}_facility_{v}")

        # Constraints: Inventory balance
        for c in range(n_components):
            model.addCons(inventory_vars[c] >= demands[c] - quicksum(transport_vars_land[f, c] + transport_vars_air[f, c] for f in range(n_facilities)), f"Inventory_{c}_Balance")

        # Constraints: Lead times and delays
        for c in range(n_components):
            model.addCons(delay_vars[c] >= lead_times[c] - quicksum(transport_vars_land[f, c] * self.lead_time_land + transport_vars_air[f, c] * self.lead_time_air for f in range(n_facilities)), f"Delay_{c}_LeadTime")
        
        # Constraints: Backorders
        for c in range(n_components):
            model.addCons(backorder_vars[c] >= demands[c] - quicksum(transport_vars_land[f, c] + transport_vars_air[f, c] for f in range(n_facilities)), f"BackOrder_{c}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value

if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_facilities': 5,
        'n_components': 100,
        'n_suppliers': 60,
        'min_prod_cost': 1500,
        'max_prod_cost': 5000,
        'min_transport_cost': 25.0,
        'max_transport_cost': 200.0,
        'min_transport_cost_air': 50.0,
        'max_transport_cost_air': 400.0,
        'min_inventory_cost': 150.0,
        'max_inventory_cost': 150.0,
        'min_lead_time': 3,
        'max_lead_time': 40,
        'lead_time_land': 0,
        'lead_time_air': 0,
        'mean_demand': 37,
        'std_dev_demand': 2,
        'min_capacity': 50,
        'max_capacity': 1500,
        'min_suppliers': 15,
        'max_suppliers': 90,
        'graph_density': 0.24,
        'bigM': 10000,
        'delay_penalty': 900.0,
        'backorder_penalty': 2500.0,
    }

    aerospace_optimizer = AerospaceSupplyChainOptimization(parameters, seed=42)
    instance = aerospace_optimizer.generate_instance()
    solve_status, solve_time, objective_value = aerospace_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")