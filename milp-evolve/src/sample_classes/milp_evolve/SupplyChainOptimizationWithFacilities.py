import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SupplyChainOptimizationWithFacilities:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_warehouses > 0 and self.n_stores > 0
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_store_cost >= 0 and self.max_store_cost >= self.min_store_cost
        assert self.min_warehouse_cap > 0 and self.max_warehouse_cap >= self.min_warehouse_cap

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        store_costs = np.random.randint(self.min_store_cost, self.max_store_cost + 1, (self.n_warehouses, self.n_stores))
        capacities = np.random.randint(self.min_warehouse_cap, self.max_warehouse_cap + 1, self.n_warehouses)
        demands = np.random.randint(1, 10, self.n_stores)
        supply_limits = np.random.uniform(self.min_supply_limit, self.max_supply_limit, self.n_warehouses)
        
        # New data generation for facility related constraints
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_costs = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        setup_costs = np.random.uniform(100, 500, size=n_facilities).tolist()
        mutual_exclusivity_pairs = [(random.randint(0, n_facilities - 1), random.randint(0, n_facilities - 1)) for _ in range(self.n_exclusive_pairs)]
        # Generating a random graph for facilities connections
        facility_graph = nx.barabasi_albert_graph(n_facilities, 3)
        graph_edges = list(facility_graph.edges)

        return {
            "warehouse_costs": warehouse_costs,
            "store_costs": store_costs,
            "capacities": capacities,
            "demands": demands,
            "supply_limits": supply_limits,
            "n_facilities": n_facilities,
            "operating_costs": operating_costs,
            "setup_costs": setup_costs,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "graph_edges": graph_edges
        }

    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        store_costs = instance['store_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        supply_limits = instance['supply_limits']
        n_facilities = instance['n_facilities']
        operating_costs = instance['operating_costs']
        setup_costs = instance['setup_costs']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        graph_edges = instance['graph_edges']

        model = Model("SupplyChainOptimizationWithFacilities")
        n_warehouses = len(warehouse_costs)
        n_stores = len(store_costs[0])

        # Decision variables
        open_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        flow_vars = {(w, s): model.addVar(vtype="C", name=f"Flow_{w}_{s}") for w in range(n_warehouses) for s in range(n_stores)}
        supply_vars = {w: model.addVar(vtype="C", name=f"Supply_{w}") for w in range(n_warehouses)}

        # New decision variables for facilities
        facility_setup_vars = {j: model.addVar(vtype="B", name=f"FacilitySetup_{j}") for j in range(n_facilities)}
        facility_usage_vars = {j: model.addVar(vtype="B", name=f"FacilityUsage_{j}") for j in range(n_facilities)}
        flow_limit_vars = {(j,s): model.addVar(vtype="C", name=f"FlowLimit_{j}_{s}") for j in range(n_facilities) for s in range(n_stores)}

        # Objective: minimize the total cost including warehouse costs, facility costs and store costs
        model.setObjective(
            quicksum(warehouse_costs[w] * open_vars[w] for w in range(n_warehouses)) +
            quicksum(store_costs[w, s] * flow_vars[w, s] for w in range(n_warehouses) for s in range(n_stores)) +
            quicksum(operating_costs[j] * facility_usage_vars[j] for j in range(n_facilities)) +
            quicksum(setup_costs[j] * facility_setup_vars[j] for j in range(n_facilities)),
            "minimize"
        )

        # Constraints: Each store's demand is met by the warehouses
        for s in range(n_stores):
            model.addCons(quicksum(flow_vars[w, s] for w in range(n_warehouses)) == demands[s], f"Store_{s}_Demand")
        
        # Constraints: Only open warehouses can supply products
        for w in range(n_warehouses):
            for s in range(n_stores):
                model.addCons(flow_vars[w, s] <= supply_limits[w] * open_vars[w], f"Warehouse_{w}_Serve_{s}")
        
        # Constraints: Warehouses cannot exceed their capacities
        for w in range(n_warehouses):
            model.addCons(quicksum(flow_vars[w, s] for s in range(n_stores)) <= capacities[w], f"Warehouse_{w}_Capacity")

        # New constraints: A facility can only process the flow if it's set up
        for j in range(n_facilities):
            for s in range(n_stores):
                model.addCons(flow_limit_vars[j, s] <= demands[s] * facility_setup_vars[j], f"Facility_{j}_FlowLimit_{s}")

        # New constraints: Ensure mutual exclusivity pairs for facilities
        for fac1, fac2 in mutual_exclusivity_pairs:
            model.addCons(facility_usage_vars[fac1] + facility_usage_vars[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        # New constraints: Ensure facility graph constraints
        for (fac1, fac2) in graph_edges:
            model.addCons(facility_usage_vars[fac1] + facility_usage_vars[fac2] <= 1, f"FacilityGraph_{fac1}_{fac2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 400,
        'n_stores': 55,
        'min_store_cost': 2250,
        'max_store_cost': 3000,
        'min_warehouse_cost': 1200,
        'max_warehouse_cost': 2500,
        'min_warehouse_cap': 375,
        'max_warehouse_cap': 1875,
        'min_supply_limit': 1000,
        'max_supply_limit': 1125,
        'facility_min_count': 20,
        'facility_max_count': 405,
        'n_exclusive_pairs': 875,
    }

    supply_optimizer = SupplyChainOptimizationWithFacilities(parameters, seed=seed)
    instance = supply_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")