import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ECommerceDeliveryNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_centers = random.randint(self.min_centers, self.max_centers)
        num_customers = random.randint(self.min_customers, self.max_customers)

        # Cost matrices
        construction_cost = np.random.randint(10000, 30000, size=num_centers)
        operational_cost = np.random.randint(500, 2000, size=num_centers)
        transport_cost = np.random.randint(1, 10, size=(num_customers, num_centers))

        # Customer demands
        customer_product_demand = np.random.randint(5, 25, size=num_customers)
        
        # Demand uncertainty
        customer_demand_uncertainty = np.random.randint(1, 3, size=num_customers)

        # Distribution centers' capacities
        center_capacity = np.random.randint(100, 500, size=num_centers)

        # Distances between centers and customers
        delivery_distance = np.random.randint(1, 20, size=(num_customers, num_centers))

        # Total budget for construction
        total_budget = np.random.randint(100000, 200000)

        res = {
            'num_centers': num_centers,
            'num_customers': num_customers,
            'construction_cost': construction_cost,
            'operational_cost': operational_cost,
            'transport_cost': transport_cost,
            'customer_product_demand': customer_product_demand,
            'customer_demand_uncertainty': customer_demand_uncertainty,
            'center_capacity': center_capacity,
            'delivery_distance': delivery_distance,
            'total_budget': total_budget
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_centers = instance['num_centers']
        num_customers = instance['num_customers']
        construction_cost = instance['construction_cost']
        operational_cost = instance['operational_cost']
        transport_cost = instance['transport_cost']
        customer_product_demand = instance['customer_product_demand']
        customer_demand_uncertainty = instance['customer_demand_uncertainty']
        center_capacity = instance['center_capacity']
        total_budget = instance['total_budget']

        model = Model("ECommerceDeliveryNetworkOptimization")

        # Variables
        CenterConstruction = {j: model.addVar(vtype="B", name=f"CenterConstruction_{j}") for j in range(num_centers)}
        ProductFlow = {(i, j): model.addVar(vtype="I", name=f"ProductFlow_{i}_{j}") for i in range(num_customers) for j in range(num_centers)}
        Inventory = {j: model.addVar(vtype="I", name=f"Inventory_{j}") for j in range(num_centers)}

        # Objective function: Minimize total costs including construction, operational, and transport costs
        TotalCost = quicksum(CenterConstruction[j] * construction_cost[j] for j in range(num_centers)) + \
                    quicksum(ProductFlow[i, j] * transport_cost[i, j] for i in range(num_customers) for j in range(num_centers)) + \
                    quicksum(Inventory[j] * operational_cost[j] for j in range(num_centers))

        model.setObjective(TotalCost, "minimize")

        # Robust customer demand constraints
        for i in range(num_customers):
            demand_min = customer_product_demand[i] - customer_demand_uncertainty[i]
            demand_max = customer_product_demand[i] + customer_demand_uncertainty[i]
            model.addCons(quicksum(ProductFlow[i, j] for j in range(num_centers)) >= demand_min, name=f"customer_demand_min_{i}")
            model.addCons(quicksum(ProductFlow[i, j] for j in range(num_centers)) <= demand_max, name=f"customer_demand_max_{i}")

        # Center capacity constraints
        for j in range(num_centers):
            model.addCons(quicksum(ProductFlow[i, j] for i in range(num_customers)) <= center_capacity[j], name=f"center_capacity_{j}")

        # Inventory constraints
        for j in range(num_centers):
            model.addCons(quicksum(ProductFlow[i, j] for i in range(num_customers)) <= Inventory[j], name=f"inventory_constraint_{j}")
        
        # Center activity constraint
        for j in range(num_centers):
            model.addCons(CenterConstruction[j] * sum(customer_product_demand) >= quicksum(ProductFlow[i, j] for i in range(num_customers)), name=f"center_activity_{j}")

        # Simplified budget constraint
        total_budget_cost = quicksum(CenterConstruction[j] * (construction_cost[j] + operational_cost[j]) for j in range(num_centers))
        model.addCons(total_budget_cost <= total_budget, name="budget_constraint")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_centers': 100,
        'max_centers': 240,
        'min_customers': 50,
        'max_customers': 3000,
    }

    optimization = ECommerceDeliveryNetworkOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")