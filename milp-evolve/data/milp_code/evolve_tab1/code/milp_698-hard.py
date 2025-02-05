import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class WarehousingDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_warehouses = random.randint(self.min_warehouses, self.max_warehouses)
        num_customers = random.randint(self.min_customers, self.max_customers)
        
        # Cost and delivery time matrices
        delivery_costs = np.random.randint(1, 100, size=(num_customers, num_warehouses))
        maintenance_costs = np.random.randint(500, 1000, size=num_warehouses)
        maintenance_budget = np.random.randint(10000, 20000)
        
        min_shipment_volumes = np.random.randint(1, 5, size=num_warehouses)
        excess_penalty = np.random.randint(10, 50)
        delivery_times = np.random.randint(1, 20, size=(num_customers, num_warehouses))

        res = {
            'num_warehouses': num_warehouses,
            'num_customers': num_customers,
            'delivery_costs': delivery_costs,
            'maintenance_costs': maintenance_costs,
            'maintenance_budget': maintenance_budget,
            'min_shipment_volumes': min_shipment_volumes,
            'excess_penalty': excess_penalty,
            'delivery_times': delivery_times,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_warehouses = instance['num_warehouses']
        num_customers = instance['num_customers']
        delivery_costs = instance['delivery_costs']
        maintenance_costs = instance['maintenance_costs']
        maintenance_budget = instance['maintenance_budget']
        min_shipment_volumes = instance['min_shipment_volumes']
        excess_penalty = instance['excess_penalty']
        delivery_times = instance['delivery_times']
        
        model = Model("WarehousingDeliveryOptimization")
        
        # Variables
        warehouse = {j: model.addVar(vtype="B", name=f"warehouse_{j}") for j in range(num_warehouses)}
        shipment = {(i, j): model.addVar(vtype="B", name=f"shipment_{i}_{j}") for i in range(num_customers) for j in range(num_warehouses)}
        excess_time = {j: model.addVar(vtype="C", name=f"excess_time_{j}") for j in range(num_warehouses)}
        
        # Objective function: Minimize total costs
        total_cost = quicksum(shipment[i, j] * delivery_costs[i, j] for i in range(num_customers) for j in range(num_warehouses)) + \
                     quicksum(warehouse[j] * maintenance_costs[j] for j in range(num_warehouses)) + \
                     quicksum(excess_time[j] * excess_penalty for j in range(num_warehouses))
        
        model.setObjective(total_cost, "minimize")
        
        # Constraints
        
        # Each customer should receive shipments from at least one warehouse
        for i in range(num_customers):
            model.addCons(quicksum(shipment[i, j] for j in range(num_warehouses)) >= 1, name=f"customer_shipment_{i}")
        
        # A warehouse can only provide shipments if it is operational
        for j in range(num_warehouses):
            for i in range(num_customers):
                model.addCons(shipment[i, j] <= warehouse[j], name=f"warehouse_shipment_{i}_{j}")
        
        # Budget constraint on the total maintenance cost
        model.addCons(quicksum(warehouse[j] * maintenance_costs[j] for j in range(num_warehouses)) <= maintenance_budget, name="maintenance_budget_constraint")
        
        # Minimum shipment volume if operational (Big M Formulation)
        for j in range(num_warehouses):
            model.addCons(warehouse[j] >= min_shipment_volumes[j] * warehouse[j], name=f"min_shipment_volume_{j}")

        # Maximum allowable delivery time constraint
        for j in range(num_warehouses):
            max_delivery_time = max(delivery_times[:, j])
            model.addCons(excess_time[j] >= max_delivery_time * warehouse[j], name=f"maximum_delivery_time_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_warehouses': 10,
        'max_warehouses': 1200,
        'min_customers': 80,
        'max_customers': 315,
    }

    optimization = WarehousingDeliveryOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")