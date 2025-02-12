import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def generate_instance(self):
        n_types_products = self.n_types_products
        
        centers_capacity = self.randint(self.n_centers, self.capacity_interval)
        centers_opening_cost = self.randint(self.n_centers, self.opening_cost_interval)
        products_demand = self.randint((self.n_customers, n_types_products), self.demand_interval)
        products_spoilage_cost = np.random.uniform(self.spoilage_cost_min, self.spoilage_cost_max, n_types_products)
        transportation_time = np.random.uniform(self.transportation_time_min, self.transportation_time_max, (self.n_centers, self.n_customers))
        temperature_requirements = np.random.uniform(self.temperature_min, self.temperature_max, n_types_products)
        storage_temperature_cost = np.random.uniform(self.temp_control_cost_min, self.temp_control_cost_max, (self.n_centers, n_types_products))
        shelf_life = np.random.randint(self.shelf_life_min, self.shelf_life_max, n_types_products)
        
        res = {
            'centers_capacity': centers_capacity,
            'centers_opening_cost': centers_opening_cost,
            'products_demand': products_demand,
            'products_spoilage_cost': products_spoilage_cost,
            'transportation_time': transportation_time,
            'temperature_requirements': temperature_requirements,
            'storage_temperature_cost': storage_temperature_cost,
            'shelf_life': shelf_life
        }
        return res

    def solve(self, instance):
        centers_capacity = instance['centers_capacity']
        centers_opening_cost = instance['centers_opening_cost']
        products_demand = instance['products_demand']
        products_spoilage_cost = instance['products_spoilage_cost']
        transportation_time = instance['transportation_time']
        temperature_requirements = instance['temperature_requirements']
        storage_temperature_cost = instance['storage_temperature_cost']
        shelf_life = instance['shelf_life']

        n_centers = len(centers_capacity)
        n_customers, n_types_products = products_demand.shape

        model = Model("SupplyChainOptimization")

        open_centers = {j: model.addVar(vtype="B", name=f"CenterOpen_{j}") for j in range(n_centers)}
        flow = {(j, k, i): model.addVar(vtype="I", name=f"Flow_{j}_{k}_{i}") for j in range(n_centers) for k in range(n_customers) for i in range(n_types_products)}
        storage_temp_cost = {(j, i): model.addVar(vtype="C", name=f"StorageTempCost_{j}_{i}") for j in range(n_centers) for i in range(n_types_products)}
        unmet_demand = {(i, k): model.addVar(vtype="I", name=f"UnmetDemand_{i}_{k}") for i in range(n_types_products) for k in range(n_customers)}

        opening_costs_expr = quicksum(centers_opening_cost[j] * open_centers[j] for j in range(n_centers))
        transport_cost_expr = quicksum(transportation_time[j, k] * flow[j, k, i] for j in range(n_centers) for k in range(n_customers) for i in range(n_types_products))
        temperature_control_cost_expr = quicksum(storage_temp_cost[j, i] for j in range(n_centers) for i in range(n_types_products))
        spoilage_penalty_expr = quicksum(products_spoilage_cost[i] * unmet_demand[i, k] for i in range(n_types_products) for k in range(n_customers))

        objective_expr = opening_costs_expr + transport_cost_expr + temperature_control_cost_expr + spoilage_penalty_expr

        for i in range(n_types_products):
            for k in range(n_customers):
                model.addCons(quicksum(flow[j, k, i] for j in range(n_centers)) + unmet_demand[i, k] == products_demand[k, i], f"Demand_{i}_{k}")

        for j in range(n_centers):
            model.addCons(quicksum(flow[j, k, i] for i in range(n_types_products) for k in range(n_customers)) <= centers_capacity[j] * open_centers[j], f"Capacity_{j}")

        for j in range(n_centers):
            for i in range(n_types_products):
                model.addCons(storage_temp_cost[j, i] >= storage_temperature_cost[j, i] * quicksum(flow[j, k, i] for k in range(n_customers)), f"TempControl_{j}_{i}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 555,
        'n_customers': 3,
        'n_types_products': 180,
        'capacity_interval': (12, 3000),
        'opening_cost_interval': (50, 10000),
        'demand_interval': (35, 350),
        'spoilage_cost_min': 500.0,
        'spoilage_cost_max': 500.0,
        'transportation_time_min': 0.1,
        'transportation_time_max': 5.62,
        'temperature_min': 0.0,
        'temperature_max': 0.75,
        'temp_control_cost_min': 450.0,
        'temp_control_cost_max': 1800.0,
        'shelf_life_min': 27,
        'shelf_life_max': 75,
    }

    supply_chain_optimization = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_chain_optimization.generate_instance()
    solve_status, solve_time = supply_chain_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")