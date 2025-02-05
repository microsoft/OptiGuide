import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ProductionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_instances(self):
        # Generating production orders and properties
        num_orders = np.random.randint(self.min_orders, self.max_orders + 1)
        num_machines = np.random.randint(self.min_machines, self.max_machines + 1)
        
        orders = []
        for i in range(num_orders):
            order_size = np.random.randint(self.min_order_size, self.max_order_size + 1)
            lead_time = np.random.randint(self.min_lead_time, self.max_lead_time + 1)
            quality_req = np.random.random()
            orders.append((order_size, lead_time, quality_req))

        machine_capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, num_machines)
        energy_usage = np.random.normal(self.mean_energy, self.std_energy, num_machines).clip(min=0.1) # Ensure no negative energy usage
        downtime_penalty = np.random.normal(self.mean_downtime_penalty, self.std_downtime_penalty, num_machines).clip(min=0.1) # Ensure no negative penalties

        res = {
            'orders': orders,
            'machine_capacities': machine_capacities,
            'energy_usage': energy_usage,
            'downtime_penalty': downtime_penalty,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        orders = instance['orders']
        machine_capacities = instance['machine_capacities']
        energy_usage = instance['energy_usage']
        downtime_penalty = instance['downtime_penalty']
        
        model = Model("ProductionOptimization")
        
        num_orders = len(orders)
        num_machines = len(machine_capacities)
        
        # Variables: x[i][j] = production quantity of order i on machine j
        x = {}
        for i in range(num_orders):
            for j in range(num_machines):
                x[i, j] = model.addVar(vtype="I", name=f"x_{i}_{j}")

        # Variables: y[j] = 1 if machine j is used, 0 otherwise
        y = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(num_machines)}
        
        # Objective: Minimize total energy consumption and production downtime
        objective = quicksum(energy_usage[j] * quicksum(x[i, j] for i in range(num_orders)) for j in range(num_machines)) + \
                    quicksum(downtime_penalty[j] * (1 - y[j]) for j in range(num_machines))
        
        # Constraints
        for i, (order_size, lead_time, quality_req) in enumerate(orders):
            model.addCons(quicksum(x[i, j] for j in range(num_machines)) >= order_size, name=f"order_{i}_fulfillment")

        for j in range(num_machines):
            model.addCons(quicksum(x[i, j] for i in range(num_orders)) <= machine_capacities[j], name=f"machine_{j}_capacity")

        for i, (order_size, lead_time, quality_req) in enumerate(orders):
            total_production_time = quicksum((1.0 / machine_capacities[j]) * x[i, j] for j in range(num_machines))
            model.addCons(total_production_time <= lead_time, name=f"order_{i}_leadtime")

        for i, (order_size, lead_time, quality_req) in enumerate(orders):
            total_quality = quicksum((np.random.random() * x[i, j]) for j in range(num_machines))
            model.addCons(total_quality >= quality_req * order_size, name=f"order_{i}_quality")

        for j in range(num_machines):
            for i in range(num_orders):
                model.addCons(x[i, j] <= y[j] * machine_capacities[j], name=f"x_{i}_{j}_binary")  # link x and y

        model.setObjective(objective, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

            
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_orders': 60,
        'max_orders': 500,
        'min_machines': 15,
        'max_machines': 100,
        'min_order_size': 400,
        'max_order_size': 500,
        'min_lead_time': 9,
        'max_lead_time': 20,
        'min_capacity': 2000,
        'max_capacity': 5000,
        'mean_energy': 60,
        'std_energy': 12,
        'mean_downtime_penalty': 250,
        'std_downtime_penalty': 500,
    }

    production_opt = ProductionOptimization(parameters, seed=seed)
    instance = production_opt.generate_instances()
    solve_status, solve_time = production_opt.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")