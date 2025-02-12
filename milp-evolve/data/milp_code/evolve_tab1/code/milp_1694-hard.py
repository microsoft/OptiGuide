import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GroceryDistributionOptimization:
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
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_warehouse_capacity > 0 and self.max_warehouse_capacity >= self.min_warehouse_capacity

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        transport_costs_edge = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_warehouses, self.n_stores))
        warehouse_capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.n_warehouses)
        grocery_demand = np.random.randint(self.demand_mean - self.demand_range, self.demand_mean + self.demand_range + 1, self.n_stores)

        G = nx.DiGraph()
        node_pairs = []
        for w in range(self.n_warehouses):
            for s in range(self.n_stores):
                G.add_edge(f"warehouse_{w}", f"store_{s}")
                node_pairs.append((f"warehouse_{w}", f"store_{s}"))

        return {
            "warehouse_costs": warehouse_costs,
            "transport_costs_edge": transport_costs_edge,
            "warehouse_capacities": warehouse_capacities,
            "grocery_demand": grocery_demand,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        transport_costs_edge = instance['transport_costs_edge']
        warehouse_capacities = instance['warehouse_capacities']
        grocery_demand = instance['grocery_demand']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("GroceryDistributionOptimization")
        n_warehouses = len(warehouse_costs)
        n_stores = len(transport_costs_edge[0])
        
        # Decision variables
        warehouse_operational_vars = {w: model.addVar(vtype="B", name=f"WarehouseOperational_{w}") for w in range(n_warehouses)}
        vehicle_warehouse_store_vars = {(u, v): model.addVar(vtype="C", name=f"VehicleWarehouseStore_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total operational and transportation costs.
        model.setObjective(
            quicksum(warehouse_costs[w] * warehouse_operational_vars[w] for w in range(n_warehouses)) +
            quicksum(transport_costs_edge[w, int(v.split('_')[1])] * vehicle_warehouse_store_vars[(u, v)] for (u, v) in node_pairs for w in range(n_warehouses) if u == f'warehouse_{w}'),
            "minimize"
        )

        # Constraints: Ensure total grocery supply matches demand at stores
        for s in range(n_stores):
            model.addCons(
                quicksum(vehicle_warehouse_store_vars[(u, f"store_{s}")] for u in G.predecessors(f"store_{s}")) >= grocery_demand[s], 
                f"Store_{s}_GroceryDemand"
            )

        # Constraints: Transport is feasible only if warehouses are operational and within maximum transport capacity
        for w in range(n_warehouses):
            for s in range(n_stores):
                model.addCons(
                    vehicle_warehouse_store_vars[(f"warehouse_{w}", f"store_{s}")] <= warehouse_operational_vars[w] * self.max_transport_capacity,
                    f"Warehouse_{w}_MaxTransportCapacity_{s}"
                )

        # Constraints: Warehouses' distribution should not exceed their capacity
        for w in range(n_warehouses):
            model.addCons(
                quicksum(vehicle_warehouse_store_vars[(f"warehouse_{w}", f"store_{s}")] for s in range(n_stores)) <= warehouse_capacities[w], 
                f"Warehouse_{w}_MaxWarehouseCapacity"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 200,
        'n_stores': 150,
        'min_transport_cost': 200,
        'max_transport_cost': 300,
        'min_warehouse_cost': 1000,
        'max_warehouse_cost': 5000,
        'min_warehouse_capacity': 2000,
        'max_warehouse_capacity': 3000,
        'max_transport_capacity': 2000,
        'demand_mean': 3000,
        'demand_range': 1500,
    }

    optimizer = GroceryDistributionOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")