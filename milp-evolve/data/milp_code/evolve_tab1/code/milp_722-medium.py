import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedOrderFulfillmentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_fulfillment_centers = random.randint(self.min_centers, self.max_centers)
        num_customers = random.randint(self.min_customers, self.max_customers)

        # Cost matrices
        order_delivery_cost = np.random.randint(50, 300, size=(num_customers, num_fulfillment_centers))
        operational_costs_center = np.random.randint(1000, 3000, size=num_fulfillment_centers)

        # Customer orders
        customer_orders = np.random.randint(100, 500, size=num_customers)

        # Fulfillment center capacities
        delivery_capacity = np.random.randint(5000, 10000, size=num_fulfillment_centers)

        res = {
            'num_fulfillment_centers': num_fulfillment_centers,
            'num_customers': num_customers,
            'order_delivery_cost': order_delivery_cost,
            'operational_costs_center': operational_costs_center,
            'customer_orders': customer_orders,
            'delivery_capacity': delivery_capacity
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_fulfillment_centers = instance['num_fulfillment_centers']
        num_customers = instance['num_customers']
        order_delivery_cost = instance['order_delivery_cost']
        operational_costs_center = instance['operational_costs_center']
        customer_orders = instance['customer_orders']
        delivery_capacity = instance['delivery_capacity']

        model = Model("SimplifiedOrderFulfillmentOptimization")

        # Variables
        new_order_fulfillment_center = {j: model.addVar(vtype="B", name=f"new_order_fulfillment_center_{j}") for j in range(num_fulfillment_centers)}
        new_delivery_path = {(i, j): model.addVar(vtype="B", name=f"new_delivery_path_{i}_{j}") for i in range(num_customers) for j in range(num_fulfillment_centers)}

        # Objective function: Minimize total costs
        total_cost = quicksum(new_delivery_path[i, j] * order_delivery_cost[i, j] for i in range(num_customers) for j in range(num_fulfillment_centers)) + \
                     quicksum(new_order_fulfillment_center[j] * operational_costs_center[j] for j in range(num_fulfillment_centers))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_customers):
            model.addCons(quicksum(new_delivery_path[i, j] for j in range(num_fulfillment_centers)) == 1, name=f"customer_delivery_{i}")

        for j in range(num_fulfillment_centers):
            for i in range(num_customers):
                model.addCons(new_delivery_path[i, j] <= new_order_fulfillment_center[j], name=f"fulfillment_center_path_{i}_{j}")

        for j in range(num_fulfillment_centers):
            model.addCons(quicksum(customer_orders[i] * new_delivery_path[i, j] for i in range(num_customers)) <= delivery_capacity[j], name=f"delivery_capacity_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_centers': 200,
        'max_centers': 500,
        'min_customers': 30,
        'max_customers': 300,
    }
    
    optimization = SimplifiedOrderFulfillmentOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")