import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedSupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_warehouses > 0 and self.n_customers > 0
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_delivery_cost >= 0 and self.max_delivery_cost >= self.min_delivery_cost
        assert self.min_warehouse_capacity > 0 and self.max_warehouse_capacity >= self.min_warehouse_capacity

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_warehouses, self.n_customers))
        capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.n_warehouses)

        product_service_prob = np.random.uniform(0.8, 1, self.n_warehouses)
        min_transport_threshold = np.random.randint(5, 15, size=(self.n_warehouses, self.n_customers))
        distribution_capacities = np.random.randint(10, 50, size=self.n_warehouses)
        
        g = nx.barabasi_albert_graph(self.n_customers, 5, seed=self.seed)

        return {
            "warehouse_costs": warehouse_costs,
            "delivery_costs": delivery_costs,
            "capacities": capacities,
            "product_service_prob": product_service_prob,
            "min_transport_threshold": min_transport_threshold,
            "distribution_capacities": distribution_capacities,
            "graph_deliveries": nx.to_numpy_array(g)
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        delivery_costs = instance['delivery_costs']
        capacities = instance['capacities']
        product_service_prob = instance['product_service_prob']
        min_transport_threshold = instance['min_transport_threshold']
        distribution_capacities = instance['distribution_capacities']
        graph_deliveries = instance['graph_deliveries']
        
        model = Model("EnhancedSupplyChainOptimization")
        n_warehouses = len(warehouse_costs)
        n_customers = len(delivery_costs[0])
        
        # Decision variables
        warehouse_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        delivery_vars = {(w, c): model.addVar(vtype="I", lb=0, ub=10, name=f"Warehouse_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}
        transport_flow_vars = {(w, c): model.addVar(vtype="C", name=f"TransportFlow_{w}_{c}") for w in range(n_warehouses) for c in range(n_customers)}

        # Objective: minimize the total cost (warehouse + delivery), apply penalties on excess usage, incentives for high service probability
        model.setObjective(quicksum(warehouse_costs[w] * warehouse_vars[w] for w in range(n_warehouses)) +
                           quicksum(delivery_costs[w][c] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)) -
                           quicksum(product_service_prob[w] * warehouse_vars[w] for w in range(n_warehouses)) +
                           quicksum(10 * (quicksum(delivery_vars[w, c] for w in range(n_warehouses)) - 1) for c in range(n_customers)), "minimize")
        
        # Constraints: Each customer must be served, but not exceeding social network capacity
        for c in range(n_customers):
            model.addCons(quicksum(delivery_vars[w, c] for w in range(n_warehouses)) == quicksum(graph_deliveries[c]))
        
        # Constraints: Only open warehouses can deliver to customers
        for w in range(n_warehouses):
            for c in range(n_customers):
                model.addCons(delivery_vars[w, c] <= 10 * warehouse_vars[w], f"Warehouse_{w}_Service_{c}")
        
        # Constraints: Warehouses cannot exceed their capacity
        for w in range(n_warehouses):
            model.addCons(quicksum(delivery_vars[w, c] for c in range(n_customers)) <= capacities[w], f"Warehouse_{w}_Capacity")

        # Additional capacity constraints for logistics
        for w in range(n_warehouses):
            model.addCons(quicksum(transport_flow_vars[w, c] for c in range(n_customers)) <= distribution_capacities[w], f"DistributionCapacity_{w}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 111,
        'n_customers': 84,
        'min_warehouse_cost': 2500,
        'max_warehouse_cost': 8000,
        'min_delivery_cost': 72,
        'max_delivery_cost': 1800,
        'min_warehouse_capacity': 21,
        'max_warehouse_capacity': 2812,
    }

    enhanced_supply_chain_optimizer = EnhancedSupplyChainOptimization(parameters, seed=42)
    instance = enhanced_supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = enhanced_supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")