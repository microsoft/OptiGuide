import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WarehouseProductDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_warehouses > 0 and self.n_stores > 0
        assert self.min_op_cost >= 0 and self.max_op_cost >= self.min_op_cost
        assert self.min_vehicle_cost >= 0 and self.max_vehicle_cost >= self.min_vehicle_cost
        assert self.min_warehouse_capacity > 0 and self.max_warehouse_capacity >= self.min_warehouse_capacity

        operation_costs = np.random.randint(self.min_op_cost, self.max_op_cost + 1, self.n_warehouses)
        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost + 1, (self.n_warehouses, self.n_stores))
        warehouse_capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.n_warehouses)
        product_requirements = np.random.randint(1, 20, self.n_stores)
        distance_limits = np.random.uniform(self.min_distance_limit, self.max_distance_limit, self.n_warehouses)
        travel_times = np.random.uniform(0, self.max_travel_time, (self.n_warehouses, self.n_stores))

        G = nx.DiGraph()
        node_pairs = []
        for w in range(self.n_warehouses):
            for s in range(self.n_stores):
                G.add_edge(f"warehouse_{w}", f"store_{s}")
                node_pairs.append((f"warehouse_{w}", f"store_{s}"))

        return {
            "operation_costs": operation_costs,
            "vehicle_costs": vehicle_costs,
            "warehouse_capacities": warehouse_capacities,
            "product_requirements": product_requirements,
            "distance_limits": distance_limits,
            "travel_times": travel_times,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        operation_costs = instance['operation_costs']
        vehicle_costs = instance['vehicle_costs']
        warehouse_capacities = instance['warehouse_capacities']
        product_requirements = instance['product_requirements']
        distance_limits = instance['distance_limits']
        travel_times = instance['travel_times']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("WarehouseProductDistribution")
        n_warehouses = len(operation_costs)
        n_stores = len(vehicle_costs[0])
        
        # Decision variables
        WarehouseProduct_vars = {w: model.addVar(vtype="B", name=f"WarehouseProduct_{w}") for w in range(n_warehouses)}
        VehicleDelivery_vars = {(u, v): model.addVar(vtype="C", name=f"VehicleDelivery_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total cost including warehouse operation costs and vehicle transportation costs.
        model.setObjective(
            quicksum(operation_costs[w] * WarehouseProduct_vars[w] for w in range(n_warehouses)) +
            quicksum(vehicle_costs[int(u.split('_')[1]), int(v.split('_')[1])] * VehicleDelivery_vars[(u, v)] for (u, v) in node_pairs),
            "minimize"
        )

        # Product distribution constraint for each store
        for s in range(n_stores):
            model.addCons(
                quicksum(VehicleDelivery_vars[(u, f"store_{s}")] for u in G.predecessors(f"store_{s}")) == product_requirements[s], 
                f"Store_{s}_DeliveryDistribution"
            )

        # Constraints: Stores only receive products if the warehouses are open and within travel distance limits
        for w in range(n_warehouses):
            for s in range(n_stores):
                model.addCons(
                    VehicleDelivery_vars[(f"warehouse_{w}", f"store_{s}")] <= distance_limits[w] * WarehouseProduct_vars[w], 
                    f"Warehouse_{w}_DistanceLimit_{s}"
                )

        # Constraints: Warehouses cannot exceed their capacities
        for w in range(n_warehouses):
            model.addCons(
                quicksum(VehicleDelivery_vars[(f"warehouse_{w}", f"store_{s}")] for s in range(n_stores)) <= warehouse_capacities[w], 
                f"Warehouse_{w}_CapacityLimit"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 150,
        'n_stores': 120,
        'min_vehicle_cost': 37,
        'max_vehicle_cost': 150,
        'min_op_cost': 250,
        'max_op_cost': 3000,
        'min_warehouse_capacity': 25,
        'max_warehouse_capacity': 100,
        'min_distance_limit': 25,
        'max_distance_limit': 150,
        'max_travel_time': 3000,
    }

    optimizer = WarehouseProductDistribution(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")