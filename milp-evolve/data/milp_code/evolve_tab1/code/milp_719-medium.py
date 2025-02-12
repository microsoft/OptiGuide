import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class OrderFulfillmentOptimization:
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
        num_periods = random.randint(self.min_periods, self.max_periods)

        # Cost matrices
        order_delivery_cost = np.random.randint(50, 300, size=(num_customers, num_fulfillment_centers))
        operational_costs_center = np.random.randint(1000, 3000, size=num_fulfillment_centers)
        
        # Customer orders
        customer_orders = {t: np.random.randint(100, 500, size=num_customers) for t in range(num_periods)}

        # Fulfillment center capacities
        delivery_capacity = np.random.randint(5000, 10000, size=num_fulfillment_centers)
        
        # Perishability, handling, storage, and penalty costs
        perishability_times = np.random.randint(1, 10, size=num_customers)
        handling_costs = np.random.randint(10, 50, size=num_fulfillment_centers)
        storage_costs = np.random.randint(5, 20, size=(num_fulfillment_centers, num_periods))
        penalty_costs = np.random.randint(20, 100, size=num_customers)

        res = {
            'num_fulfillment_centers': num_fulfillment_centers,
            'num_customers': num_customers,
            'order_delivery_cost': order_delivery_cost,
            'operational_costs_center': operational_costs_center,
            'customer_orders': customer_orders,
            'delivery_capacity': delivery_capacity,
            'perishability_times': perishability_times,
            'handling_costs': handling_costs,
            'storage_costs': storage_costs,
            'penalty_costs': penalty_costs,
            'num_periods': num_periods,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_fulfillment_centers = instance['num_fulfillment_centers']
        num_customers = instance['num_customers']
        num_periods = instance['num_periods']
        order_delivery_cost = instance['order_delivery_cost']
        operational_costs_center = instance['operational_costs_center']
        customer_orders = instance['customer_orders']
        delivery_capacity = instance['delivery_capacity']
        perishability_times = instance['perishability_times']
        handling_costs = instance['handling_costs']
        storage_costs = instance['storage_costs']
        penalty_costs = instance['penalty_costs']

        model = Model("OrderFulfillmentOptimization")

        # Variables
        new_order_fulfillment_center = {j: model.addVar(vtype="B", name=f"new_order_fulfillment_center_{j}") for j in range(num_fulfillment_centers)}
        new_delivery_path = {(i, j): model.addVar(vtype="B", name=f"new_delivery_path_{i}_{j}") for i in range(num_customers) for j in range(num_fulfillment_centers)}
        delivery_time = {(i, j): model.addVar(vtype="I", name=f"delivery_time_{i}_{j}") for i in range(num_customers) for j in range(num_fulfillment_centers)}
        penalty_cost = {(i, j): model.addVar(vtype="C", name=f"penalty_cost_{i}_{j}") for i in range(num_customers) for j in range(num_fulfillment_centers)}
        storage_cost = {(j, t): model.addVar(vtype="C", name=f"storage_cost_{j}_{t}") for j in range(num_fulfillment_centers) for t in range(num_periods)}
        
        # Objective function: Minimize total costs including handling, storage, and penalty costs
        total_cost = quicksum(new_delivery_path[i, j] * order_delivery_cost[i, j] for i in range(num_customers) for j in range(num_fulfillment_centers)) + \
                     quicksum(new_order_fulfillment_center[j] * operational_costs_center[j] for j in range(num_fulfillment_centers)) + \
                     quicksum(handling_costs[j] * new_order_fulfillment_center[j] for j in range(num_fulfillment_centers)) + \
                     quicksum(storage_cost[j, t] for j in range(num_fulfillment_centers) for t in range(num_periods)) + \
                     quicksum(penalty_cost[i, j] for i in range(num_customers) for j in range(num_fulfillment_centers))
        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_customers):
            for j in range(num_fulfillment_centers):
                # Ensure delivery time does not exceed perishability time
                model.addCons(delivery_time[i, j] <= perishability_times[i], name=f"perishability_{i}_{j}")
                # Penalty cost if delivery time exceeds a certain threshold (e.g., perishability time)
                model.addCons(penalty_cost[i, j] >= delivery_time[i, j] * penalty_costs[i], name=f"penalty_{i}_{j}")

        for i in range(num_customers):
            model.addCons(quicksum(new_delivery_path[i, j] for j in range(num_fulfillment_centers)) == 1, name=f"customer_delivery_{i}")

        for j in range(num_fulfillment_centers):
            for i in range(num_customers):
                model.addCons(new_delivery_path[i, j] <= new_order_fulfillment_center[j], name=f"fulfillment_center_path_{i}_{j}")

        for j in range(num_fulfillment_centers):
            for t in range(num_periods):
                model.addCons(quicksum(customer_orders[t][i] * new_delivery_path[i, j] for i in range(num_customers)) <= 
                              new_order_fulfillment_center[j] * delivery_capacity[j], name=f"capacity_{j}_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_centers': 25,
        'max_centers': 100,
        'min_customers': 45,
        'max_customers': 450,
        'min_periods': 15,
        'max_periods': 60,
    }
    optimization = OrderFulfillmentOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")