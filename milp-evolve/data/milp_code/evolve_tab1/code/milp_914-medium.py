import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SupplyChainWithSetCovering:
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

        g = nx.barabasi_albert_graph(self.n_customers, 5, seed=self.seed)
        
        # Generate random customer sets for set covering
        customer_sets = []
        num_sets = np.random.randint(2, 6)
        for _ in range(num_sets):
            set_size = np.random.randint(2, max(5, self.n_customers//4))
            customer_sets.append(np.random.choice(self.n_customers, set_size, replace=False).tolist())
        
        return {
            "warehouse_costs": warehouse_costs,
            "delivery_costs": delivery_costs,
            "capacities": capacities,
            "graph_deliveries": nx.to_numpy_array(g),
            "customer_sets": customer_sets
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        delivery_costs = instance['delivery_costs']
        capacities = instance['capacities']
        graph_deliveries = instance['graph_deliveries']
        customer_sets = instance['customer_sets']
        
        model = Model("SupplyChainWithSetCovering")
        n_warehouses = len(warehouse_costs)
        n_customers = len(delivery_costs[0])
        
        # Decision variables
        warehouse_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        delivery_vars = {(w, c): model.addVar(vtype="I", lb=0, ub=10, name=f"Warehouse_{w}_Customer_{c}") for w in range(n_warehouses) for c in range(n_customers)}
        set_service_vars = {(w, s): model.addVar(vtype="B", name=f"Warehouse_{w}_Set_{s}") for w in range(n_warehouses) for s in range(len(customer_sets))}

        # Objective: minimize the total cost (warehouse + delivery)
        model.setObjective(quicksum(warehouse_costs[w] * warehouse_vars[w] for w in range(n_warehouses)) +
                           quicksum(delivery_costs[w][c] * delivery_vars[w, c] for w in range(n_warehouses) for c in range(n_customers)), "minimize")
        
        # Constraints: Each customer must be served, but not exceeding social network capacity
        for c in range(n_customers):
            model.addCons(quicksum(delivery_vars[w, c] for w in range(n_warehouses)) == quicksum(graph_deliveries[c]), f"CustomerDemand_{c}")
        
        # Constraints: Only open warehouses can deliver to customers
        for w in range(n_warehouses):
            for c in range(n_customers):
                model.addCons(delivery_vars[w, c] <= 10 * warehouse_vars[w], f"Warehouse_{w}_Service_{c}")
        
        # Constraints: Warehouses cannot exceed their capacity
        for w in range(n_warehouses):
            model.addCons(quicksum(delivery_vars[w, c] for c in range(n_customers)) <= capacities[w], f"Warehouse_{w}_Capacity")
        
        # Set covering constraints: Each customer set must be covered by one warehouse
        for s, customer_set in enumerate(customer_sets):
            for w in range(n_warehouses):
                for c in customer_set:
                    model.addCons(delivery_vars[w, c] >= set_service_vars[w, s], f"Warehouse_{w}_SetService_{c}_Set_{s}")
            
            model.addCons(quicksum(set_service_vars[w, s] for w in range(n_warehouses)) == 1, f"SetCovering_{s}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 222,
        'n_customers': 31,
        'min_warehouse_cost': 625,
        'max_warehouse_cost': 8000,
        'min_delivery_cost': 54,
        'max_delivery_cost': 675,
        'min_warehouse_capacity': 945,
        'max_warehouse_capacity': 2109,
        'num_customer_sets': 2,
    }

    supply_chain_optimizer = SupplyChainWithSetCovering(parameters, seed=42)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")