import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class WarehouseDistributionOptimization:
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
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_warehouses, self.n_stores))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_warehouses)
        store_demands = np.random.randint(1, 10, self.n_stores)
        equipment_limits = np.random.uniform(self.min_equipment_limit, self.max_equipment_limit, self.n_warehouses)
        distances = np.random.uniform(0, self.max_transport_distance, (self.n_warehouses, self.n_stores))

        # Create a flow network graph using NetworkX
        G = nx.DiGraph()
        node_pairs = []
        for w in range(self.n_warehouses):
            for s in range(self.n_stores):
                G.add_edge(f"warehouse_{w}", f"store_{s}")
                node_pairs.append((f"warehouse_{w}", f"store_{s}"))
                
        return {
            "warehouse_costs": warehouse_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "store_demands": store_demands,
            "equipment_limits": equipment_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        store_demands = instance['store_demands']
        equipment_limits = instance['equipment_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("WarehouseDistributionOptimization")
        n_warehouses = len(warehouse_costs)
        n_stores = len(transport_costs[0])
        
        # Decision variables
        open_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        flow_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total cost including warehouse operating costs and transport costs.
        model.setObjective(
            quicksum(warehouse_costs[w] * open_vars[w] for w in range(n_warehouses)) +
            quicksum(transport_costs[w, int(v.split('_')[1])] * flow_vars[(u, v)] for (u, v) in node_pairs for w in range(n_warehouses) if u == f'warehouse_{w}'),
            "minimize"
        )

        # Flow conservation constraint for each store
        for s in range(n_stores):
            model.addCons(
                quicksum(flow_vars[(u, f"store_{s}")] for u in G.predecessors(f"store_{s}")) == store_demands[s], 
                f"Store_{s}_NodeFlowConservation"
            )

        # Constraints: Warehouses only send flows if they are open
        for w in range(n_warehouses):
            for s in range(n_stores):
                model.addCons(
                    flow_vars[(f"warehouse_{w}", f"store_{s}")] <= equipment_limits[w] * open_vars[w], 
                    f"Warehouse_{w}_FlowLimitByEquipment_{s}"
                )

        # Constraints: Warehouses cannot exceed their storage capacities
        for w in range(n_warehouses):
            model.addCons(
                quicksum(flow_vars[(f"warehouse_{w}", f"store_{s}")] for s in range(n_stores)) <= capacities[w], 
                f"Warehouse_{w}_MaxWarehouseCapacity"
            )

        # Movement distance constraint (Critical store)
        for s in range(n_stores):
            model.addCons(
                quicksum(open_vars[w] for w in range(n_warehouses) if distances[w, s] <= self.max_transport_distance) >= 1, 
                f"Store_{s}_CriticalWarehouseCoverage"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 70,
        'n_stores': 135,
        'min_transport_cost': 125,
        'max_transport_cost': 600,
        'min_warehouse_cost': 800,
        'max_warehouse_cost': 937,
        'min_capacity': 937,
        'max_capacity': 1000,
        'min_equipment_limit': 1080,
        'max_equipment_limit': 2136,
        'max_transport_distance': 1050,
    }
    
    warehouse_optimizer = WarehouseDistributionOptimization(parameters, seed=seed)
    instance = warehouse_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")