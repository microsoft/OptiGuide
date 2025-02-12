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
        # Randomly set the number of centers and customers
        num_fulfillment_centers = random.randint(self.min_centers, self.max_centers)
        num_customers = random.randint(self.min_customers, self.max_customers)

        # Costs and capacities
        operational_costs_center = np.random.randint(1000, 3000, size=num_fulfillment_centers)
        customer_demand = np.random.randint(150, 450, size=num_customers)
        center_capacities = np.random.randint(6000, 12000, size=num_fulfillment_centers)

        instance_data = {
            'num_fulfillment_centers': num_fulfillment_centers,
            'num_customers': num_customers,
            'operational_costs_center': operational_costs_center,
            'customer_demand': customer_demand,
            'center_capacities': center_capacities,
        }
        return instance_data

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_fulfillment_centers = instance['num_fulfillment_centers']
        num_customers = instance['num_customers']
        operational_costs_center = instance['operational_costs_center']
        customer_demand = instance['customer_demand']
        center_capacities = instance['center_capacities']

        # Create Optimization Model
        model = Model("Simplified_OrderFulfillmentOptimization")

        # Variables
        use_center = {j: model.addVar(vtype="B", name=f"use_center_{j}") for j in range(num_fulfillment_centers)}
        allocate_demand = {(i, j): model.addVar(vtype="C", name=f"allocate_demand_{i}_{j}") for i in range(num_customers) for j in range(num_fulfillment_centers)}
        
        # Objective function: Minimize operational costs
        total_cost = quicksum(use_center[j] * operational_costs_center[j] for j in range(num_fulfillment_centers))

        model.setObjective(total_cost, "minimize")

        # Constraints
        # Capacity Constraints: Ensure no center exceeds its capacity
        for j in range(num_fulfillment_centers):
            model.addCons(quicksum(allocate_demand[i, j] for i in range(num_customers)) <= use_center[j] * center_capacities[j], name=f"capacity_{j}")

        # Demand Constraints: Ensure all customer demands are met
        for i in range(num_customers):
            model.addCons(quicksum(allocate_demand[i, j] for j in range(num_fulfillment_centers)) == customer_demand[i], name=f"demand_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_centers': 120,
        'max_centers': 700,
        'min_customers': 100,
        'max_customers': 2400,
    }

    optimization = OrderFulfillmentOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")